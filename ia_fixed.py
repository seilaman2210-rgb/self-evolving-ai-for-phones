"""
╔══════════════════════════════════════════════════════════════════════╗
║  🧬  SELF-EVOLVING AI  v4  —  MoE + CHECKPOINTS + PREMIUM SKILLS  🧬 ║
╠══════════════════════════════════════════════════════════════════════╣
║  Novidades v4:                                                       ║
║   ✅ Pasta premium/ — um checkpoint por habilidade distinta          ║
║      Critérios: reward >= 0.80, complexidade real, stdout novo       ║
║      Nome legível: skill_001_r1.00_gen42_hello_world.pt              ║
║      Limite configurável (PREMIUM_MAX_SKILLS = 100)                  ║
║      Retomada entre sessões: não re-salva habilidades já existentes  ║
║  Mantido de v3:                                                      ║
║   ✅ Dedup por STDOUT                                                ║
║   ✅ Pós-choque corrigido                                            ║
║   ✅ Supervised bloqueado em stdout repetido                         ║
║   ✅ Tecla P [Enter] = novo pré-treino sem reiniciar                 ║
║  Mantido de v2:                                                      ║
║   ✅ Mixture of Experts (MoE)                                        ║
║   ✅ Checkpoints automáticos + melhor modelo                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Instalar:                                                           ║
║    pip install torch rich tqdm                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess, sys, time, random, math, os, threading, queue, select
from collections import deque

from rich.live    import Live
from rich.panel   import Panel
from rich.columns import Columns
from rich.text    import Text
from rich.console import Console, Group
from rich         import box

console = Console()

# ══════════════════════════════════════════════════════════════════════
#  ⌨️  LEITOR DE COMANDOS (background thread — tecla P para pré-treino)
# ══════════════════════════════════════════════════════════════════════
_cmd_queue: "queue.Queue[str]" = queue.Queue()

def _stdin_watcher():
    """
    Thread daemon: monitora stdin sem bloquear o loop principal.
    Comandos suportados (digitar + Enter):
      P          → novo pré-treino com PRETRAIN_STEPS padrão
      P 5000     → novo pré-treino com 5000 steps
    """
    while True:
        try:
            r, _, _ = select.select([sys.stdin], [], [], 1.0)
            if r:
                line = sys.stdin.readline().strip()
                if line:
                    _cmd_queue.put(line)
        except Exception:
            break

_stdin_thread = threading.Thread(target=_stdin_watcher, daemon=True)
_stdin_thread.start()

# ══════════════════════════════════════════════════════════════════════
#  ⚙️  CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════
SEED             = 42
CONTEXT_LEN      = 64
PRETRAIN_STEPS   = 3000
PRETRAIN_LR      = 3e-3
PRETRAIN_LR_CONT = 8e-4   # LR de continuação — menor para não sobrescrever o que RL aprendeu
REINFORCE_LR     = 5e-4
SUPERVISED_LR    = 1e-3
MAX_CODE_LEN     = 64
CODE_TIMEOUT     = 3
MUTATION_EVERY   = 300
MUTATION_TRIALS  = 3
MUTATION_STEPS   = 80
REWARD_WINDOW    = 30
TEMP_START       = 1.1
TEMP_MIN         = 0.6
TEMP_DECAY       = 0.9997
ENTROPY_COEF     = 0.03

# ── Detecção de entropia travada → pré-treino automático ─────────────
# Se a entropia ficar alta (modelo gerando tokens aleatórios) E o reward
# médio continuar baixo por muitas gerações seguidas, o modelo está preso
# num equilíbrio ruim (ex: spammar comentários vale quase tanto quanto
# tentar código real). Solução: pré-treino automático para re-ancorar.
ENTROPY_STUCK_THRESHOLD = 2.40   # entropia acima disso = distribuição quase uniforme
ENTROPY_STUCK_REWARD    = 0.25   # avg reward abaixo disso = modelo não está aprendendo
ENTROPY_STUCK_PATIENCE  = 30     # gerações consecutivas antes de disparar
ENTROPY_STUCK_PT_STEPS  = 3000   # steps de pré-treino de re-ancoragem

# ── Checkpoint ────────────────────────────────────────────────────────
CHECKPOINT_DIR   = "./checkpoints"           # pasta onde ficam os .pt
CHECKPOINT_EVERY = 50                        # salva a cada N gerações
CHECKPOINT_LAST  = "checkpoint_last.pt"      # checkpoint mais recente
CHECKPOINT_BEST  = "checkpoint_best.pt"      # melhor reward até agora

# ── Premium Skills ─────────────────────────────────────────────────────
# Pasta onde ficam os checkpoints "premium": um por habilidade distinta.
# Critérios para entrar:
#   1. reward >= PREMIUM_MIN_REWARD
#   2. stdout genuinamente novo (não normaliza para algo já salvo)
#   3. código com lógica real (complexity_factor >= PREMIUM_MIN_FACTOR)
# O total é limitado a PREMIUM_MAX_SKILLS para não encher o disco.
PREMIUM_DIR        = "./checkpoints/premium"
PREMIUM_MIN_REWARD = 0.80    # reward mínimo para considerar "boa"
PREMIUM_MIN_FACTOR = 0.60    # fator de complexidade mínimo (filtra print("x") trivial)
PREMIUM_MAX_SKILLS = 100     # máximo de arquivos na pasta premium

# ── MoE ───────────────────────────────────────────────────────────────
# use_moe=True  → cada bloco usa Mixture-of-Experts no lugar do FFN simples
# n_experts     → total de "especialistas" (sub-redes FFN) por bloco
# top_k         → quantos especialistas são ativados por token (sparse routing)
INIT_CONFIG = dict(
    embed_dim = 96,
    n_heads   = 4,
    n_layers  = 3,
    dropout   = 0.1,
    use_moe   = True,
    n_experts = 2,      # menos experts = menos peso parado na RAM do celular
    top_k     = 1,
)
# Config "mobile": ~0.6M parâmetros, CONTEXT_LEN=128, batch pequeno (ver
# get_batch). Prioriza rodar de boa no Termux/Pydroid3 em vez de bater
# 10M+ params — troque esses números se depois for treinar num PC/GPU.

# ── Aprendizado a partir de código externo ────────────────────────────
LEARN_SUP_STEPS    = 400
LEARN_RL_TRIES     = 300
LEARN_REPS_NEEDED  = 40
LEARN_TEMP_START   = 1.0
EXPAND_EMBED_DELTA = 16
MAX_CODE_LEN_LEARN = 120

torch.manual_seed(SEED)
random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════
#  📚  CORPUS DE PRÉ-TREINO  (todos os .py no diretório)
# ══════════════════════════════════════════════════════════════════════

def carregar_corpus_recursivo(diretorios_raiz, extensoes=(".py",)):
    if isinstance(diretorios_raiz, str):
        diretorios_raiz = [diretorios_raiz]
    conteudo_total = ""
    for diretorio_raiz in diretorios_raiz:
        if not os.path.isdir(diretorio_raiz):
            continue   # ex: /kaggle/input não existe fora do Kaggle — ignora de boa
        for pasta_atual, subpastas, arquivos in os.walk(diretorio_raiz):
            for nome_arquivo in arquivos:
                if not nome_arquivo.endswith(extensoes):
                    continue
                if nome_arquivo in ("self_evolving_ai-1.py", "self_evolving_ai_v2.py"):
                    continue
                caminho_completo = os.path.join(pasta_atual, nome_arquivo)
                try:
                    with open(caminho_completo, "r", encoding="utf-8", errors="ignore") as f:
                        conteudo_total += f.read() + "\n"
                        print(f"Lido: {nome_arquivo}")
                except Exception as e:
                    print(f"Erro ao ler {nome_arquivo}: {e}")
    return conteudo_total

diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
# Datasets adicionados via "Add Data" no Kaggle são montados (read-only) em
# /kaggle/input/<nome-do-dataset>/. Fora do Kaggle essa pasta não existe e é
# ignorada automaticamente (ver checagem os.path.isdir acima).
# .txt entra pra dar espaço a datasets de texto puro (contos, letras, etc),
# além dos .py de sempre. CSV não é lido direto — teria coluna/estrutura
# própria, teria que extrair a coluna de texto antes e salvar como .txt.
EXTRA_CORPUS_DIRS = ["/kaggle/input"]
CORPUS = carregar_corpus_recursivo([diretorio_raiz] + EXTRA_CORPUS_DIRS,
                                    extensoes=(".py", ".txt"))

# ══════════════════════════════════════════════════════════════════════
#  🔤  TOKENIZER  (character-level)
# ══════════════════════════════════════════════════════════════════════
EOS   = "\x00"
chars = sorted(set(CORPUS + EOS))
VOCAB = len(chars)

c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}

encode = lambda s: [c2i[c] for c in s if c in c2i]
decode = lambda l: "".join(i2c.get(i, "?") for i in l)

data_tensor = torch.tensor(encode(CORPUS), dtype=torch.long)

# ══════════════════════════════════════════════════════════════════════
#  🖥️  DISPOSITIVO
# ══════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    dev_str  = f"🚀 {gpu_name} ({vram_gb:.1f}GB VRAM)"
else:
    device  = torch.device("cpu")
    dev_str = "⚠️  CPU (sem GPU detectada)"
    # Usa todos os núcleos disponíveis para os kernels de matmul do PyTorch.
    # Sem isso o PyTorch às vezes fica preso em 1 thread e desperdiça o
    # resto da CPU à toa — impacto direto em tokens/segundo.
    try:
        _n_cpu = os.cpu_count() or 1
        torch.set_num_threads(_n_cpu)
        torch.set_num_interop_threads(max(1, _n_cpu // 2))
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════
#  🏗️  ARQUITETURA
# ══════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Atenção causal com suporte a KV-CACHE.

    Antes: cada novo caractere gerado recomputava a atenção sobre TODA a
    sequência desde o início (O(n^2) por amostra, e O(n^3) no total de uma
    geração de n chars). Isso é o principal motivo do modelo ser lento em
    tokens/segundo. Agora, past_kv guarda as chaves/valores já computados
    e cada passo novo só processa o(s) token(s) novo(s), reaproveitando o
    resto — geração ~O(n) em vez de O(n^2).
    """
    def __init__(self, embed_dim, n_heads, dropout, ctx):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.hd      = embed_dim // n_heads
        self.qkv     = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out     = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ad      = nn.Dropout(dropout)
        self.rd      = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(ctx, ctx)).view(1, 1, ctx, ctx)
        self.register_buffer("mask", mask)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        split = lambda t: t.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if use_cache else None

        Tk = k.shape[2]
        w = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        if past_kv is None:
            # Sem cache (treino/avaliação em batch): máscara causal normal.
            w = w.masked_fill(self.mask[:, :, :T, :Tk] == 0, float("-inf"))
        # Com cache: q contém só o(s) token(s) novo(s), que por definição
        # vêm depois de tudo que já está no cache → podem atender a tudo,
        # não precisa de máscara.
        w = self.ad(F.softmax(w, dim=-1))
        o = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.rd(self.out(o)), new_kv


# ══════════════════════════════════════════════════════════════════════
#  🧩  MIXTURE OF EXPERTS  (novo em v2)
# ══════════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """
    Substitui o FFN denso por N 'especialistas' independentes.

    Como funciona:
      1. Router (Linear + Softmax) decide o peso de cada especialista
         para cada token.
      2. Apenas top_k especialistas são ativados (sparse routing) —
         os outros recebem peso 0 e NÃO são computados → eficiência.
      3. A saída é a soma ponderada das saídas dos especialistas ativos.

    Vantagem: com 4 experts top-2, o modelo tem 4x mais "conhecimento"
    de FFN mas só usa 2x o custo computacional por forward pass.
    Especialistas tendem a se especializar em padrões distintos de código
    (ex: um em print(), outro em def, outro em imports, etc.).
    """

    def __init__(self, embed_dim: int, n_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = min(top_k, n_experts)

        # Roteador: para cada token, produz logits sobre os N experts
        self.router = nn.Linear(embed_dim, n_experts, bias=False)

        # N FFN idênticos na estrutura mas com pesos independentes
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)               # (N_tok, C)

        # ── Roteamento ────────────────────────────────────────────────
        router_logits = self.router(x_flat)          # (N_tok, n_experts)
        router_probs  = F.softmax(router_logits, dim=-1)

        # Seleciona os top_k experts para cada token
        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)
        # Renormaliza para que os pesos dos top_k somem 1
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # ── Cálculo dos experts (sparse) ──────────────────────────────
        output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            # Máscara: quais tokens rotearam para este expert?
            # topk_idx shape: (N_tok, top_k)
            is_selected = (topk_idx == expert_id)          # (N_tok, top_k) bool
            token_mask  = is_selected.any(dim=-1)           # (N_tok,) bool
            if not token_mask.any():
                continue

            # Pesos deste expert para cada token selecionado
            # Quando um token ativa este expert em múltiplas posições top_k
            # (impossível com top_k únicos, mas seguro assim mesmo):
            weight = is_selected[token_mask].float() * topk_probs[token_mask]
            weight = weight.sum(dim=-1, keepdim=True)       # (n_selected, 1)

            expert_out = expert(x_flat[token_mask])          # (n_selected, C)
            output[token_mask] += expert_out * weight

        return output.reshape(B, T, C)

    def expert_load(self) -> torch.Tensor:
        """Retorna fração de uso esperada por expert (para diagnóstico)."""
        return torch.ones(self.n_experts) / self.n_experts   # placeholder estático


class Block(nn.Module):
    """Bloco Transformer com atenção causal + FFN ou MoE."""

    def __init__(self, embed_dim, n_heads, dropout, ctx,
                 use_moe=False, n_experts=4, top_k=2):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout, ctx)
        self.ln2  = nn.LayerNorm(embed_dim)

        if use_moe:
            self.mlp = MoELayer(embed_dim, n_experts=n_experts,
                                top_k=top_k, dropout=dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout),
            )

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TinyAI(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        ed      = cfg["embed_dim"]
        nh      = cfg["n_heads"]
        nl      = cfg["n_layers"]
        do      = cfg["dropout"]
        use_moe = cfg.get("use_moe", False)
        ne      = cfg.get("n_experts", 4)
        tk      = cfg.get("top_k", 2)

        self.tok_emb = nn.Embedding(VOCAB, ed)
        self.pos_emb = nn.Embedding(CONTEXT_LEN, ed)
        self.drop    = nn.Dropout(do)
        # ModuleList (não Sequential) porque cada bloco agora recebe/retorna
        # o cache de KV além do tensor — Sequential só encadeia um único valor.
        self.blocks  = nn.ModuleList([
            Block(ed, nh, do, CONTEXT_LEN, use_moe=use_moe, n_experts=ne, top_k=tk)
            for _ in range(nl)
        ])
        self.ln_f    = nn.LayerNorm(ed)
        self.head    = nn.Linear(ed, VOCAB, bias=False)
        self.apply(self._init)
        self._cfg = cfg

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        B, T = idx.shape
        past_len = past_kv[0][0].shape[2] if past_kv is not None else 0
        pos  = torch.arange(past_len, past_len + T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            x, nkv = block(x, past_kv=pkv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(nkv)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = (F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
                if targets is not None else None)
        return logits, loss, new_kvs

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════
#  💾  CHECKPOINTS  (novo em v2)
# ══════════════════════════════════════════════════════════════════════

def _ckpt_path(filename: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, filename)


def save_checkpoint(model: TinyAI, rl_opt, sup_opt, state: dict,
                    filename: str = CHECKPOINT_LAST,
                    pretrain_opt=None) -> None:
    """
    Salva um snapshot completo:
      - Pesos do modelo (state_dict)
      - Estado dos três otimizadores (AdamW moments — inclui pretrain_opt)
      - Configuração da arquitetura (para reconstruir o modelo)
      - Estado de treino (geração, melhor reward, temperatura, histórico…)
      - Vocabulário completo (chars) → permite detectar/corrigir mismatch de vocab
    """
    path = _ckpt_path(filename)
    payload = {
        # ── Modelo ────────────────────────────────────────────────────
        "model_cfg"       : model._cfg,
        "model_state"     : model.state_dict(),
        # ── Vocabulário (salvo para detectar mudanças no corpus) ──────
        "vocab_chars"     : chars,          # lista ordenada de caracteres
        "vocab_size"      : VOCAB,
        # ── Otimizadores ──────────────────────────────────────────────
        "rl_opt_state"    : rl_opt.state_dict(),
        "sup_opt_state"   : sup_opt.state_dict(),
        "pretrain_opt_state": pretrain_opt.state_dict() if pretrain_opt is not None else None,
        # ── Estado de treino ──────────────────────────────────────────
        "gen"             : state["gen"],
        "temp"            : state["temp"],
        "best_reward"     : state["best_reward"],
        "best_code"       : state["best_code"],
        "best_gen"        : state["best_gen"],
        "config"          : state["config"],
        "mutation_log"    : state["mutation_log"],
        "reward_hist"     : list(state["reward_hist"]),
        "temp_resets"     : state["temp_resets"],
        "stdout_memory"   : list(state.get("stdout_memory", [])),
        "stdout_norm_mem" : list(state.get("stdout_norm_memory", [])),
    }
    torch.save(payload, path)


def load_checkpoint(filename: str = CHECKPOINT_LAST):
    """
    Carrega checkpoint se existir.
    Retorna o dicionário salvo, ou None se não houver arquivo.
    """
    path = _ckpt_path(filename)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt



# ══════════════════════════════════════════════════════════════════════
#  🌟  PREMIUM SKILLS  —  uma habilidade distinta por arquivo
# ══════════════════════════════════════════════════════════════════════

def _premium_slug(stdout: str, code: str) -> str:
    """
    Gera um slug legível para nomear o arquivo premium.
    Usa os primeiros 30 chars do stdout limpo, com underscores.
    Ex: stdout='Hello, World!' → 'hello_world'
         stdout='0\n1\n2'      → '0_1_2'
    """
    import re as _re2
    slug = stdout.strip()[:40]
    slug = _re2.sub(r'[^a-zA-Z0-9 ]', '', slug)   # só alfanumérico e espaço
    slug = slug.strip().lower()
    slug = _re2.sub(r'\s+', '_', slug)             # espaços → _
    slug = slug[:30] or "skill"
    return slug


def save_premium_checkpoint(
    model: "TinyAI",
    rl_opt, sup_opt, pretrain_opt,
    state: dict,
    code: str,
    stdout: str,
    r_val: float,
    gen: int,
    *,
    premium_stdout_set: set,
) -> bool:
    """
    Salva um checkpoint na pasta premium SE a habilidade for nova e boa.

    Critérios (todos devem ser satisfeitos):
      1. reward >= PREMIUM_MIN_REWARD
      2. fator de complexidade do código >= PREMIUM_MIN_FACTOR
      3. stdout normalizado ainda não presente em premium_stdout_set
         (garante que cada entrada é uma habilidade *distinta*)
      4. pasta premium não excedeu PREMIUM_MAX_SKILLS arquivos

    Retorna True se salvou, False caso contrário.
    """
    import re as _re2

    # ── Critério 1: reward mínimo ──────────────────────────────────────
    if r_val < PREMIUM_MIN_REWARD:
        return False

    # ── Critério 2: código com lógica real ────────────────────────────
    factor = _code_complexity_factor(code)
    if factor < PREMIUM_MIN_FACTOR:
        return False

    # ── Critério 3: stdout normalizado é novo no premium ──────────────
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = _re2.sub(r'\s+', ' ', s)
        s = _re2.sub(r'[^a-z0-9 ]', '', s)
        s = _re2.sub(r'(.)\1{2,}', r'\1', s)
        return s.strip()

    norm_out = _norm(stdout) if stdout else ""
    if norm_out in premium_stdout_set:
        return False

    # ── Critério 4: limite de arquivos ────────────────────────────────
    os.makedirs(PREMIUM_DIR, exist_ok=True)
    existing = [f for f in os.listdir(PREMIUM_DIR) if f.endswith(".pt")]
    if len(existing) >= PREMIUM_MAX_SKILLS:
        return False

    # ── Tudo ok: salva ────────────────────────────────────────────────
    premium_stdout_set.add(norm_out)

    slug      = _premium_slug(stdout, code)
    seq       = len(existing) + 1
    filename  = f"skill_{seq:03d}_r{r_val:.2f}_gen{gen}_{slug}.pt"
    path      = os.path.join(PREMIUM_DIR, filename)

    payload = {
        # ── Modelo ────────────────────────────────────────────────────
        "model_cfg"         : model._cfg,
        "model_state"       : model.state_dict(),
        # ── Vocabulário ───────────────────────────────────────────────
        "vocab_chars"       : chars,
        "vocab_size"        : VOCAB,
        # ── Otimizadores ──────────────────────────────────────────────
        "rl_opt_state"      : rl_opt.state_dict(),
        "sup_opt_state"     : sup_opt.state_dict(),
        "pretrain_opt_state": pretrain_opt.state_dict() if pretrain_opt else None,
        # ── Habilidade ────────────────────────────────────────────────
        "skill_code"        : code,
        "skill_stdout"      : stdout,
        "skill_reward"      : r_val,
        "skill_gen"         : gen,
        "skill_complexity"  : factor,
        # ── Estado global no momento do save ─────────────────────────
        "gen"               : state["gen"],
        "temp"              : state["temp"],
        "best_reward"       : state["best_reward"],
        "best_code"         : state["best_code"],
        "best_gen"          : state["best_gen"],
        "config"            : state["config"],
        "mutation_log"      : state["mutation_log"],
        "reward_hist"       : list(state["reward_hist"]),
        "temp_resets"       : state["temp_resets"],
        "stdout_memory"     : list(state.get("stdout_memory", [])),
        "stdout_norm_mem"   : list(state.get("stdout_norm_memory", [])),
    }
    torch.save(payload, path)
    return True


def _init_premium_stdout_set() -> set:
    """
    Lê os checkpoints já existentes na pasta premium e devolve o conjunto
    de stdouts normalizados já salvos — evita duplicatas ao retomar treino.
    """
    import re as _re2

    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = _re2.sub(r'\s+', ' ', s)
        s = _re2.sub(r'[^a-z0-9 ]', '', s)
        s = _re2.sub(r'(.)\1{2,}', r'\1', s)
        return s.strip()

    seen = set()
    if not os.path.isdir(PREMIUM_DIR):
        return seen
    for fname in os.listdir(PREMIUM_DIR):
        if not fname.endswith(".pt"):
            continue
        try:
            ckpt = torch.load(
                os.path.join(PREMIUM_DIR, fname),
                map_location="cpu",
                weights_only=False,
            )
            out = ckpt.get("skill_stdout", "")
            if out:
                seen.add(_norm(out))
        except Exception:
            pass
    return seen


def _transplant_state_dict(saved_sd: dict, model: "TinyAI") -> None:
    """
    Carrega `saved_sd` em `model` copiando os pesos compatíveis e
    ignorando diferenças de shape (ex: vocab mudou → tok_emb e head
    têm tamanhos diferentes).  Tokens novos ficam com init aleatório.
    Igual a transplant_weights() mas opera diretamente em state_dicts.
    """
    dst_sd = model.state_dict()
    for key in dst_sd:
        if key not in saved_sd:
            continue
        s, d = saved_sd[key], dst_sd[key]
        if s.shape == d.shape:
            dst_sd[key] = s.clone()
        else:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(s.shape, d.shape))
            dst_sd[key][slices] = s[slices].clone()
    model.load_state_dict(dst_sd)


def restore_from_checkpoint(ckpt: dict, state: dict):
    """
    Reconstrói model + otimizadores a partir do checkpoint
    e atualiza o dicionário `state` in-place.
    Retorna (model, rl_opt, sup_opt, pretrain_opt, vocab_mismatch).

    Tolerante a mismatch de vocab: se o corpus cresceu/encolheu
    (novos .py no diretório adicionam chars ao vocabulário), usa
    _transplant_state_dict para copiar pesos compatíveis em vez de
    explodir com RuntimeError de shape mismatch em tok_emb/head.
    Quando isso ocorre, retorna vocab_mismatch=True para que main()
    possa rodar um pré-treino de recuperação antes de entrar no RL.
    """
    model = TinyAI(ckpt["model_cfg"]).to(device)

    # Detecta vocab do checkpoint pelo shape real do tensor (funciona mesmo
    # em checkpoints antigos que não salvavam a chave "vocab_size")
    ckpt_vocab = ckpt["model_state"]["tok_emb.weight"].shape[0]
    vocab_mismatch = (ckpt_vocab != VOCAB)

    if vocab_mismatch:
        # ── Vocab mudou: transplante parcial ──────────────────────────
        delta = VOCAB - ckpt_vocab
        console.print(
            f"[bold yellow]⚠️  Vocab mudou: checkpoint={ckpt_vocab} chars  "
            f"atual={VOCAB} chars  ({delta:+d})[/bold yellow]\n"
            f"   → Transplantando pesos compatíveis; "
            f"{abs(delta)} token(s) {'novo(s)' if delta > 0 else 'removido(s)'} "
            f"ficam com init aleatório.\n"
            f"   → Pré-treino de recuperação será executado automaticamente."
        )
        _transplant_state_dict(ckpt["model_state"], model)
        # Otimizadores recriados do zero (moments incompatíveis com novo vocab)
        rl_opt       = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt      = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR,
                                          weight_decay=1e-4)
    else:
        # ── Vocab idêntico: restaura tudo exatamente ──────────────────
        model.load_state_dict(ckpt["model_state"])
        rl_opt  = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        rl_opt.load_state_dict(ckpt["rl_opt_state"])
        sup_opt.load_state_dict(ckpt["sup_opt_state"])

        # Restaura pretrain_opt (pode não existir em checkpoints antigos)
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR,
                                          weight_decay=1e-4)
        pt_state = ckpt.get("pretrain_opt_state")
        if pt_state is not None:
            try:
                pretrain_opt.load_state_dict(pt_state)
            except Exception:
                pass  # Checkpoint antigo sem pretrain_opt — começa do zero só o opt

    # Restaura estado de treino
    state["gen"]          = ckpt["gen"]
    state["temp"]         = ckpt["temp"]
    state["best_reward"]  = ckpt["best_reward"]
    state["best_code"]    = ckpt["best_code"]
    state["best_gen"]     = ckpt["best_gen"]
    state["config"]       = ckpt["config"]
    state["mutation_log"] = ckpt["mutation_log"]
    state["temp_resets"]  = ckpt["temp_resets"]
    state["reward_hist"].extend(ckpt.get("reward_hist", []))
    state["stdout_memory"].update(ckpt.get("stdout_memory", []))
    state["stdout_norm_memory"].update(ckpt.get("stdout_norm_mem", []))
    state["n_params"]     = model.n_params

    return model, rl_opt, sup_opt, pretrain_opt, vocab_mismatch


# ══════════════════════════════════════════════════════════════════════
#  🎲  GERAÇÃO RL
# ══════════════════════════════════════════════════════════════════════

def generate_rl(model, temperature=0.9, on_char=None):
    """
    Gera código char a char com gradientes ativos (para REINFORCE).
    Retorna: (código_gerado, log_probs_tensor, entropies_tensor)

    FIX DE PERFORMANCE: a versão anterior recomputava a atenção sobre TODA
    a sequência gerada até agora a CADA novo caractere (cond = tokens[-256:]
    → forward completo de novo). Isso é custo quadrático em MAX_CODE_LEN e
    era o principal gargalo de tokens/segundo do treino.
    Agora usa o KV-cache do model(): o primeiro passo processa o token
    inicial, e cada passo seguinte processa SÓ o token novo, reaproveitando
    as chaves/valores já computados — custo linear em vez de quadrático.
    """
    model.train()
    log_probs    = []
    entropies    = []
    chars_so_far = []
    tok      = torch.tensor([[c2i.get("\n", 0)]], dtype=torch.long, device=device)
    past_kv  = None

    for _ in range(MAX_CODE_LEN):
        logits, _, past_kv = model(tok, past_kv=past_kv, use_cache=True)
        logits    = logits[:, -1, :] / max(temperature, 0.1)
        probs     = F.softmax(logits, dim=-1)
        dist      = torch.distributions.Categorical(probs)
        sampled   = dist.sample()
        lp        = dist.log_prob(sampled)
        ent       = -(probs * (probs + 1e-8).log()).sum()

        log_probs.append(lp)
        entropies.append(ent)

        ch = i2c.get(sampled.item(), "?")
        chars_so_far.append(ch)
        if on_char:
            on_char("".join(chars_so_far))

        full = "".join(chars_so_far)
        if EOS in full or full.count("\n") >= 2:
            break

        tok = sampled.unsqueeze(0)   # próximo forward só recebe o token novo

    code  = "".join(chars_so_far).strip().replace(EOS, "")
    lp_t  = torch.stack(log_probs)  if log_probs  else None
    ent_t = torch.stack(entropies)  if entropies  else None
    return code, lp_t, ent_t


# ══════════════════════════════════════════════════════════════════════
#  🧪  EXECUÇÃO SEGURA EM SUBPROCESS
# ══════════════════════════════════════════════════════════════════════

def safe_exec(code: str):
    if not code.strip():
        return -3, "", "código vazio"
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=CODE_TIMEOUT,
        )
        return r.returncode, r.stdout.strip()[:200], r.stderr.strip()[:200]
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)[:100]


# ══════════════════════════════════════════════════════════════════════
#  🏆  RECOMPENSA
# ══════════════════════════════════════════════════════════════════════

import re as _re

# ── Helper: detecta código sem nenhuma linha executável ───────────────
def _is_only_comments(code: str) -> bool:
    """
    Retorna True se o código não tem NENHUMA linha executável
    (só comentários, linhas em branco e strings solitárias).

    Isso fecha o exploit onde '# [lixo aleatório]' gera rc=0,
    stdout='' e recebia reward 0.25 por ser 'código limpo sem output'.
    """
    for line in code.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            return False
    return True


# Palavras-chave que indicam código com lógica real
_COMPLEXITY_KW = [
    'for ', 'while ', 'if ', 'elif ', 'else:',
    'def ', 'class ', 'import ', 'from ',
    'return ', 'yield ', 'range(', 'len(',
    'sum(', 'list(', 'dict(', 'set(', 'map(',
    'lambda ', 'try:', 'except ', 'with ',
    'open(', 'math.', 'random.',
]

# Regex: detecta linha que é só print("literal") ou print('literal') ou print(número)
_TRIVIAL_PRINT_RE = _re.compile(
    r'^print\s*\(\s*(?:["\'][^"\']*["\']|\d+)\s*\)$'
)

# Regex: detecta atribuição simples (x = ..., _var = ...)
_SIMPLE_ASSIGN_RE = _re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+$')

# Keywords que indicam REAL estrutura de controle (exclui import/lambda
# pois gerar `import os` sem output não merece bônus)
_REAL_FLOW_KW = [
    'for ', 'while ', 'if ', 'elif ', 'else:',
    'def ', 'class ',
    'return ', 'yield ', 'range(', 'len(',
    'sum(', 'list(', 'dict(', 'set(', 'map(',
    'try:', 'except ', 'with ',
    'open(', 'math.', 'random.',
]

# Keywords "fracas" — presença não garante complexidade real
_WEAK_KW = ['import ', 'from ', 'lambda ']

def _code_complexity_factor(code: str) -> float:
    """
    Fator multiplicador de reward baseado na estrutura real do código.
    Fechamos cada exploit por categoria:

      0.02  — só comentários / linhas em branco
      0.10  — só atribuições simples (x=0, proor=0) — sem possibilidade de output
      0.15  — TODOS os prints (qualquer argumento), sem fluxo real
              Fecha: print(True), print(1+1), print('a'*3), print(len('x'))
      0.50  — mix de atribuições + prints sem fluxo
      0.80  — outras linhas sem keywords de fluxo
      1.00  — tem import/lambda mas sem fluxo real (bônus mínimo)
      1.40  — tem for/if/def/while/try/etc. (fluxo real)
    """
    lines = [l.strip() for l in code.split('\n')
             if l.strip() and not l.strip().startswith('#')]
    if not lines:
        return 0.02

    has_real_flow = any(kw in code for kw in _REAL_FLOW_KW)
    has_weak_kw   = any(kw in code for kw in _WEAK_KW)

    print_lines     = [l for l in lines if l.startswith('print')]
    non_print_lines = [l for l in lines if not l.startswith('print')]

    # Caso 1: tem fluxo real → bônus máximo
    if has_real_flow:
        return 1.40

    # Caso 2: só imports / lambdas sem output (import os, lambda x: x)
    # Eram falsos positivos com factor=1.40 → bônus mínimo
    if has_weak_kw and not print_lines:
        return 0.80

    # Caso 3: TODOS os prints (qualquer argumento), sem fluxo
    # Fecha: print(True), print(1+1), print("a"*3), print(len("x"))
    # Antes: print não-literal tinha factor=0.50 → reward 0.35 (acima da baseline)
    if print_lines and not non_print_lines:
        return 0.15

    # Caso 4: APENAS atribuições simples, sem prints, sem fluxo
    # Fecha: x=0; proor=0 → completamente inútil, sem caminho para output
    if not print_lines and all(_SIMPLE_ASSIGN_RE.match(l) for l in lines):
        return 0.10

    # Caso 5: mix de atribuições + prints, sem fluxo
    if print_lines and non_print_lines:
        return 0.50

    # Caso 6: outros (expressões soltas, pass, ellipsis, etc.)
    return 0.80


def reward(rc, stdout, stderr, code: str = "") -> float:
    """
    Recompensa baseada em: execução limpa + output + complexidade do código.

    Escala:
      TIMEOUT / vazio              →  0.00
      SyntaxError                  →  0.05
      Erro de runtime              →  0.10–0.20
      Roda, sem output             →  0.15–0.70  (varia com complexidade)
      Roda, output genérico        →  0.10–0.98  (trivial print → ~0.10)
      Roda, output = saudação      →  1.00–2.00
    """
    if rc in (-1, -3):
        return 0.0
    # ── Exploit do comentário: '# [lixo]' → rc=0, stdout='' ─────────
    # Código sem linhas executáveis não merece nenhum reward positivo.
    if code and _is_only_comments(code):
        return 0.01
    if rc != 0:
        # Erros de runtime ainda encorajam tentar (sintaxe correta > erros)
        factor = _code_complexity_factor(code) if code else 1.0
        base   = 0.05 if "SyntaxError" in stderr else 0.20
        return round(min(base * factor, base), 3)

    # ── Código rodou com sucesso ──────────────────────────────────────
    factor = _code_complexity_factor(code) if code else 1.0

    if not stdout:
        # Código que roda mas não produz nada não está progredindo em direção
        # ao objetivo (produzir output). Reward baixo o suficiente para ficar
        # SEMPRE abaixo da baseline — o modelo não deve ser recompensado por
        # gerar `x = 0`, `proor = 0`, etc. indefinidamente.
        #
        # Escala atual com o fix:
        #   atribuição simples   (factor=0.80) → 0.12 * 0.80 = 0.10
        #   código com for/if    (factor=1.40) → 0.12 * 1.40 = 0.17
        #   máximo possível                   → 0.20
        # Baseline típica após algumas gerações: ~0.20–0.40 → sempre abaixo.
        return round(min(0.12 * factor, 0.20), 3)

    sl        = stdout.lower().strip()
    canonical = {"hi", "hello", "hey", "hi there", "hello world", "hey there"}

    if sl in canonical:
        # Saudação canônica: recompensa máxima independente de complexidade
        # (queremos que o modelo chegue nisso de formas variadas)
        return 2.0

    if any(w in sl for w in ("hi", "hello", "hey")):
        return round(1.0 * max(factor, 0.5), 3)   # mínimo 0.5 mesmo trivial

    # Output genérico — aqui que o exploit vivia (0.7 sempre)
    # Agora: print("here") → 0.7 * 0.15 ≈ 0.10
    #        for i in range(3): print(i) → 0.7 * 1.4 ≈ 0.98
    return round(min(0.70 * factor, 0.98), 3)


# ══════════════════════════════════════════════════════════════════════
#  🎓  PRÉ-TREINO SUPERVISIONADO
# ══════════════════════════════════════════════════════════════════════

def get_batch(bs=4):
    max_i = len(data_tensor) - CONTEXT_LEN - 1
    if max_i <= 0: return None, None
    ix = torch.randint(0, max_i, (bs,))
    x  = torch.stack([data_tensor[i: i + CONTEXT_LEN] for i in ix])
    y  = torch.stack([data_tensor[i + 1: i + CONTEXT_LEN + 1] for i in ix])
    return x.to(device), y.to(device)


def pretrain(model, steps, lr, status_callback=None, opt=None):
    """
    Treino supervisionado no corpus.

    Se `opt` for passado, reutiliza o otimizador existente — os momentos
    (média e variância do AdamW) acumulados nas rodadas anteriores são
    preservados, então o modelo *continua* de onde parou em vez de
    recomeçar do zero.  O LR do otimizador é atualizado para `lr`.

    Se `opt` for None (padrão), cria um AdamW novo — usado apenas para
    modelos descartáveis como em eval_config() / eval_mutation_from_model().
    """
    created_here = opt is None
    if created_here:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        # Atualiza o LR sem destruir os momentos acumulados
        for pg in opt.param_groups:
            pg["lr"] = lr

    for step in range(steps):
        x, y = get_batch()
        if x is None: break
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if status_callback:
            status_callback(step, steps, loss.item())
    return model


# ══════════════════════════════════════════════════════════════════════
#  📈  REINFORCE UPDATE  (com entropy regularization)
# ══════════════════════════════════════════════════════════════════════

def reinforce_update(model, optimizer, log_probs, entropies, r_val, baseline):
    if log_probs is None or entropies is None or len(log_probs) == 0:
        return 0.0, 0.0
    advantage     = r_val - baseline
    policy_loss   = -log_probs.sum() * advantage
    entropy_bonus = -ENTROPY_COEF * entropies.mean()
    loss          = policy_loss + entropy_bonus
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), entropies.mean().item()


def supervised_on_success(model, optimizer, code, steps=20):
    toks = encode(code + "\n")
    if len(toks) < 4: return
    t = torch.tensor(toks, dtype=torch.long, device=device)
    for _ in range(steps):
        x = t[:-1].unsqueeze(0)
        y = t[1:].unsqueeze(0)
        if x.shape[1] < 1: break
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


# ══════════════════════════════════════════════════════════════════════
#  🔀  TRANSPLANTE DE PESOS ENTRE ARQUITETURAS  (corrigido em v2)
# ══════════════════════════════════════════════════════════════════════

def transplant_weights(src: TinyAI, dst: TinyAI) -> None:
    """
    Copia pesos compatíveis de `src` para `dst`.
    Quando as dimensões diferem (ex: após mutação de embed_dim),
    copia apenas a parte que cabe — o resto fica com init aleatório.
    Isso "herda" o aprendizado do modelo anterior mesmo com nova forma.
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd:
        if key not in src_sd:
            continue
        s, d = src_sd[key], dst_sd[key]
        if s.shape == d.shape:
            dst_sd[key] = s.clone()
        else:
            # Copia a região de interseção dimensional
            slices = tuple(slice(0, min(a, b)) for a, b in zip(s.shape, d.shape))
            dst_sd[key][slices] = s[slices].clone()
    dst.load_state_dict(dst_sd)


# ══════════════════════════════════════════════════════════════════════
#  🧬  MUTAÇÃO DE ARQUITETURA (NAS simplificado)
# ══════════════════════════════════════════════════════════════════════

def mutate_config(cfg: dict) -> dict:
    new  = cfg.copy()
    kind = random.choice(["embed", "layers", "heads", "dropout", "experts"])
    if kind == "embed":
        new["embed_dim"] = max(32, cfg["embed_dim"] + random.choice([-16, 16, 32]))
        new["embed_dim"] = (new["embed_dim"] // new["n_heads"]) * new["n_heads"]
    elif kind == "layers":
        new["n_layers"] = max(1, min(8, cfg["n_layers"] + random.choice([-1, 1])))
    elif kind == "heads":
        new["n_heads"]   = random.choice([2, 4, 8])
        new["embed_dim"] = max(new["n_heads"] * 8, new["embed_dim"])
        new["embed_dim"] = (new["embed_dim"] // new["n_heads"]) * new["n_heads"]
    elif kind == "dropout":
        new["dropout"] = round(max(0.0, min(0.4, cfg["dropout"] + random.choice([-0.05, 0.05]))), 2)
    elif kind == "experts" and cfg.get("use_moe"):
        # Muta número de experts (potência de 2 entre 2 e 8)
        choices = [e for e in [2, 4, 8] if e != cfg.get("n_experts", 4)]
        new["n_experts"] = random.choice(choices)
        new["top_k"]     = min(new["n_experts"] - 1, cfg.get("top_k", 2))
    return new


def _eval_model_loss(model, n_batches=20):
    """Avalia cross-entropy média do modelo no corpus."""
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(16)
            if x is None: break
            _, l, _ = model(x, y)
            losses.append(l.item())
    return sum(losses) / max(1, len(losses))


def eval_config(cfg: dict):
    """
    Constrói modelo com `cfg`, treina brevemente e avalia.
    Retorna (model, loss_media).  Corrigido em v2 (ausente no v1).
    """
    m = TinyAI(cfg).to(device)
    pretrain(m, MUTATION_STEPS, PRETRAIN_LR)
    return m, _eval_model_loss(m)


def eval_mutation_from_model(base_model: TinyAI, cfg: dict, steps: int = MUTATION_STEPS):
    """
    Constrói modelo com `cfg`, transplanta pesos de `base_model`,
    treina brevemente e avalia.  Retorna (model, loss_media).
    """
    m = TinyAI(cfg).to(device)
    transplant_weights(base_model, m)
    pretrain(m, steps, PRETRAIN_LR)
    return m, _eval_model_loss(m)


# ══════════════════════════════════════════════════════════════════════
#  🖥️  DISPLAY AO VIVO  (rich)
# ══════════════════════════════════════════════════════════════════════

REWARD_COLORS = {
    0.0: "red", 0.05: "red", 0.2: "orange3",
    0.5: "yellow", 0.7: "green", 1.0: "bright_green", 2.0: "bold magenta",
}

def reward_color(r_val):
    for thresh in sorted(REWARD_COLORS.keys(), reverse=True):
        if r_val >= thresh:
            return REWARD_COLORS[thresh]
    return "white"

def sparkline(values, width=28):
    BLOCKS = "▁▂▃▄▅▆▇█"
    if not values: return "─" * width
    mx = max(values) or 1
    return "".join(BLOCKS[min(7, int(v / mx * 7))] for v in list(values)[-width:])

def make_display(st):
    cfg       = st["config"]
    hist      = st["reward_hist"]
    avg_r     = sum(list(hist)[-20:]) / max(1, min(20, len(hist)))
    phase_lbl = "[bold cyan]PRÉ-TREINO[/]" if st["phase"] == "pretrain" else "[bold green]EVOLUINDO[/]"

    div       = st.get("diversity", 1.0)
    div_color = "bright_green" if div >= 0.5 else ("yellow" if div >= 0.3 else "bold red blink")
    div_str   = f"[{div_color}]{div:.0%}[/]"

    ent       = st.get("last_entropy", 0.0)
    ent_color = "bright_green" if ent > 2.0 else ("yellow" if ent > 1.0 else "bold red")

    stuck     = st.get("entropy_stuck_count", 0)
    stuck_str = (f"  │  [bold red blink]🆘 EntStuck:{stuck}/{ENTROPY_STUCK_PATIENCE}[/]"
                 if stuck > ENTROPY_STUCK_PATIENCE // 2 else "")

    ckpt_str  = f"💾 ckpt@gen{st.get('last_ckpt_gen', '—')}" if st.get('last_ckpt_gen') else "💾 sem ckpt"

    header = Panel(
        f"{phase_lbl}  │  "
        f"Gen: [bold]{st['gen']}[/]  │  "
        f"Melhor: [bold {reward_color(st['best_reward'])}]{st['best_reward']:.2f}[/]  │  "
        f"Avg(20): [bold]{avg_r:.2f}[/]  │  "
        f"Div: {div_str}  │  "
        f"Entropia: [{ent_color}]{ent:.2f}[/]  │  "
        f"Temp: {st['temp']:.3f}  │  "
        f"Resets: {st.get('temp_resets', 0)}  │  "
        f"{ckpt_str}  │  "
        f"[dim]⌨️  P [+steps] = pré-treino[/dim]  │  "
        f"{dev_str}"
        f"{stuck_str}",
        style="bold blue", box=box.HEAVY
    )

    code_str  = st["current_code"] or "[dim]aguardando...[/dim]"
    code_text = Text(code_str, style="bold green")
    if st["phase"] == "generate":
        code_text.append("█", style="blink bright_green")
    code_panel = Panel(code_text, title="[bold green]🖊️  GERANDO CÓDIGO (ao vivo)[/]",
                       border_style="green", box=box.ROUNDED)

    rc, stdout, stderr = st.get("last_exec", (None, "", ""))
    if rc is None:
        exec_body = "[dim]nenhuma execução ainda[/dim]"
    elif rc == 0:
        exec_body = (f"[bold green]✅ returncode: 0[/]\n"
                     f"stdout: [bright_white]{repr(stdout[:60])}[/]\n"
                     f"stderr: [dim]—[/dim]")
    else:
        tag = {-1: "TIMEOUT ⏱", -3: "VAZIO"}.get(rc, f"ERRO (rc={rc})")
        exec_body = (f"[bold red]❌ {tag}[/]\n"
                     f"stdout: [dim]{repr(stdout[:40])}[/]\n"
                     f"stderr: [yellow]{stderr[:80]}[/yellow]")
    last_r = st.get("last_reward", 0.0)
    exec_panel = Panel(
        exec_body + f"\n[bold]reward: [{reward_color(last_r)}]{last_r:.2f}[/][/bold]",
        title="[bold yellow]⚡ EXECUÇÃO[/]", border_style="yellow", box=box.ROUNDED
    )

    best_body = (
        f"[bold cyan]{st['best_code'] or 'nenhum ainda...'}[/]\n"
        f"reward: [bold {reward_color(st['best_reward'])}]{st['best_reward']:.2f}[/]  │  "
        f"na geração {st['best_gen']}"
    )
    best_panel = Panel(best_body, title="[bold cyan]🏆 MELHOR RESULTADO[/]",
                       border_style="cyan", box=box.ROUNDED)

    spark     = sparkline(list(hist))
    recent    = list(hist)[-10:]
    dist_str  = " ".join(f"[{reward_color(r)}]{r:.1f}[/]" for r in recent)
    hist_panel = Panel(
        f"[bold]{spark}[/]\n{dist_str}",
        title="[bold magenta]📊 HISTÓRICO DE REWARDS[/]",
        border_style="magenta", box=box.ROUNDED
    )

    # ── Painel da arquitetura mostra info do MoE ──────────────────────
    gens_to_mut = MUTATION_EVERY - (st["gen"] % MUTATION_EVERY)
    mut_log     = "  │  ".join(st["mutation_log"][-3:]) if st["mutation_log"] else "—"
    moe_info    = ""
    if cfg.get("use_moe"):
        moe_info = (f"  │  [bold]MoE[/]: experts={cfg.get('n_experts', 4)} "
                    f"top-k={cfg.get('top_k', 2)}")
    arch_panel = Panel(
        f"embed={cfg['embed_dim']}  layers={cfg['n_layers']}  heads={cfg['n_heads']}  "
        f"dropout={cfg['dropout']}{moe_info}  │  params={st['n_params']:,}  │  "
        f"próxima mutação em: [bold]{gens_to_mut}[/] gens\n"
        f"hist: {mut_log}",
        title="[bold blue]🧬 ARQUITETURA (MoE + auto-evolução)[/]",
        border_style="blue", box=box.ROUNDED
    )

    log_lines = "\n".join(st["log"][-5:]) or "[dim]...[/dim]"
    log_panel = Panel(log_lines, title="[dim]📝 LOG[/dim]", border_style="dim",
                      box=box.SIMPLE)

    return Group(
        header,
        Columns([code_panel, exec_panel], equal=True),
        Columns([best_panel, hist_panel], equal=True),
        arch_panel,
        log_panel,
    )


# ══════════════════════════════════════════════════════════════════════
#  🖥️  HELPER: ATUALIZAÇÃO THROTTLED DO DISPLAY
# ══════════════════════════════════════════════════════════════════════

def make_throttled_updater(live, state, min_interval: float = 0.10):
    """
    Retorna uma função `update()` que só chama live.update() se passaram
    pelo menos `min_interval` segundos desde a última chamada.

    Por que isso importa:
      Rich's Live renderiza o painel inteiro a cada chamada. No Windows/CPU,
      se o terminal não conseguir acompanhar (buffer cheio, resize, etc.),
      a chamada pode bloquear por dezenas de ms — travando o loop principal.
      Throttling por tempo garante no máximo ~10 renders/s independente de
      quantas vezes update() é chamada.
    """
    _last = [0.0]
    def update(force: bool = False):
        now = time.monotonic()
        if force or now - _last[0] >= min_interval:
            live.update(make_display(state))
            _last[0] = now
    return update


# ══════════════════════════════════════════════════════════════════════
#  🚀  MAIN — LOOP DE AUTO-EVOLUÇÃO
# ══════════════════════════════════════════════════════════════════════

def main():
    console.print("\n[bold blue]══ 🧬 SELF-EVOLVING AI v4  (MoE + Checkpoints + Premium Skills) ══[/bold blue]")
    console.print(f"   {dev_str}")
    console.print(f"   vocab={VOCAB} chars  |  context={CONTEXT_LEN}  |  corpus={len(data_tensor)} tokens\n")

    # ── Estado compartilhado ──────────────────────────────────────────
    state = dict(
        phase         = "pretrain",
        gen           = 0,
        temp          = TEMP_START,
        config        = INIT_CONFIG.copy(),
        n_params      = 0,
        current_code  = "",
        last_exec     = (None, "", ""),
        last_reward   = 0.0,
        best_code     = "",
        best_reward   = 0.0,
        best_gen      = 0,
        reward_hist   = deque(maxlen=REWARD_WINDOW * 3),
        mutation_log  = [],
        log           = [],
        code_memory   = set(),
        stdout_memory = set(),     # ← dedup por saída exata
        stdout_norm_memory = set(), # ← dedup por saída normalizada (pega variações triviais)
        recent_codes  = deque(maxlen=30),
        diversity     = 0.0,
        last_entropy  = 0.0,
        temp_resets   = 0,
        last_ckpt_gen = None,      # ← rastreia último checkpoint
        # ── Detecção de entropia travada ──────────────────────────────
        entropy_stuck_count = 0,   # gerações consecutivas com entropia alta + reward baixo
        # ── Premium skills ────────────────────────────────────────────
        premium_count       = 0,   # quantas habilidades premium já salvas nesta sessão
    )

    # Carrega stdouts já salvos na pasta premium (persistência entre sessões)
    premium_stdout_set = _init_premium_stdout_set()
    if premium_stdout_set:
        console.print(f"[bold magenta]🌟 Premium: {len(premium_stdout_set)} habilidade(s) já salva(s) em '{PREMIUM_DIR}'[/bold magenta]")

    def log(msg):
        state["log"].append(msg)

    # ── Tenta carregar checkpoint ─────────────────────────────────────
    ckpt = load_checkpoint(CHECKPOINT_LAST)
    if ckpt is not None:
        console.print("[bold yellow]♻️  Checkpoint encontrado! Retomando treino...[/bold yellow]")
        model, rl_opt, sup_opt, pretrain_opt, vocab_mismatch = restore_from_checkpoint(ckpt, state)
        state["last_ckpt_gen"] = state["gen"]
        log(f"♻️  Retomado do checkpoint: gen={state['gen']}  best_reward={state['best_reward']:.2f}")
        # Vocab mudou → precisa de pré-treino de recuperação para re-estabilizar
        # os pesos aleatórios dos tokens novos antes de entrar em RL
        skip_pretrain = not vocab_mismatch
        if vocab_mismatch:
            log(f"⚠️  Vocab mudou → pré-treino de recuperação agendado")
        else:
            # Mesmo sem vocab mismatch, se a sessão anterior estava numa
            # situação ruim (reward baixo = modelo corrompido por RL), agenda
            # um pré-treino curto de re-ancoragem antes de continuar RL.
            avg_hist = (sum(ckpt.get("reward_hist", [0])[-20:]) /
                        max(1, min(20, len(ckpt.get("reward_hist", [0])))))
            if avg_hist < ENTROPY_STUCK_REWARD:
                skip_pretrain  = False
                vocab_mismatch = True  # reutiliza a flag para acionar pt de recuperação
                log(f"⚠️  Sessão anterior com avg_reward={avg_hist:.2f} < {ENTROPY_STUCK_REWARD} "
                    f"→ pré-treino de re-ancoragem agendado")
    else:
        console.print("[bold green]🆕 Nenhum checkpoint encontrado. Começando do zero.[/bold green]")
        model        = TinyAI(INIT_CONFIG).to(device)
        state["n_params"] = model.n_params
        rl_opt       = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
        sup_opt      = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR,
                                          weight_decay=1e-4)
        skip_pretrain = False
        vocab_mismatch = False
        log(f"Modelo MoE criado: {model.n_params:,} params  (experts={INIT_CONFIG['n_experts']} top-k={INIT_CONFIG['top_k']})")

    with Live(make_display(state), refresh_per_second=10, screen=False) as live:
        # refresh() = atualização throttled (máx 10/s) — evita travar o terminal
        # refresh(force=True) = atualização imediata p/ mudanças de fase importantes
        refresh = make_throttled_updater(live, state, min_interval=0.10)

        # ══ FASE 1: PRÉ-TREINO  (novo ou recuperação pós-vocab-mismatch) ════
        if not skip_pretrain:
            state["phase"] = "pretrain"
            # Pré-treino de recuperação usa menos steps (só re-estabiliza tokens novos)
            pt_steps = 2500 if vocab_mismatch else PRETRAIN_STEPS
            pt_label = "recuperação" if vocab_mismatch else "inicial"
            log(f"Iniciando pré-treino de {pt_label} ({pt_steps} steps)...")
            refresh(force=True)

            def pretrain_cb(step, total, loss):
                state["current_code"] = f"[pré-treino {pt_label}] step {step}/{total}  loss={loss:.4f}"
                refresh()
                if step > 0 and step % 50 == 0:
                    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)

            pretrain(model, pt_steps, PRETRAIN_LR, pretrain_cb, opt=pretrain_opt)
            log(f"✅ Pré-treino de {pt_label} concluído. Iniciando auto-evolução...")

        state["phase"]        = "evolve"
        state["current_code"] = ""
        refresh(force=True)
        time.sleep(0.3)

        # ══ FASE 2: LOOP DE AUTO-EVOLUÇÃO ════════════════════════════
        try:
            while True:
                state["gen"] += 1
                gen = state["gen"]
                state["phase"] = "generate"

                def on_char(s):
                    state["current_code"] = s
                    refresh()   # throttled — no máx 10x/s, não a cada char

                code, log_probs, entropies = generate_rl(
                    model, temperature=state["temp"], on_char=on_char
                )
                state["current_code"] = code
                state["phase"]        = "exec"
                refresh(force=True)

                # ── Execução ──────────────────────────────────────────
                rc, stdout, stderr = safe_exec(code)
                r_val = reward(rc, stdout, stderr, code)

                state["last_exec"]   = (rc, stdout, stderr)
                state["last_reward"] = r_val
                state["reward_hist"].append(r_val)

                # ── Deduplicação por CÓDIGO (hash exato) ──────────────
                code_hash    = hash(code)
                is_duplicate = code_hash in state["code_memory"]
                state["code_memory"].add(code_hash)
                state["recent_codes"].append(code)
                unique_recent    = len(set(state["recent_codes"]))
                state["diversity"] = unique_recent / max(1, len(state["recent_codes"]))

                # ── Deduplicação por STDOUT (exato + normalizado) ─────
                # "stdout_memory" guarda strings exatas (exato dedup).
                # "stdout_norm_memory" guarda forma normalizada para pegar
                # variações triviais: "here" / "herere" / "HERE" / " here "
                # → todas normalizam para "here" e são tratadas como iguais.
                def _normalize_stdout(s: str) -> str:
                    s = s.lower().strip()
                    # Colapsa espaços/newlines
                    s = _re.sub(r'\s+', ' ', s)
                    # Remove caracteres não-alfanuméricos (pontuação, emojis)
                    s = _re.sub(r'[^a-z0-9 ]', '', s)
                    # Colapsa sequências de chars repetidos: "heeeere" → "here"
                    s = _re.sub(r'(.)\1{2,}', r'\1', s)
                    # Remove sufixos/prefixos repetidos: "herere" → "here"
                    # (detecta repetição de substring de comprimento >= 2)
                    for seg_len in range(len(s) // 2, 1, -1):
                        seg = s[:seg_len]
                        if s == seg * (len(s) // seg_len) and len(s) % seg_len == 0:
                            s = seg
                            break
                    return s.strip()

                stdout_key      = stdout.strip() if stdout else ""
                stdout_norm_key = _normalize_stdout(stdout_key) if stdout_key else ""

                stdout_is_new = (
                    stdout_key == ""
                    or (stdout_key      not in state["stdout_memory"])
                    and (stdout_norm_key not in state["stdout_norm_memory"])
                )
                if stdout_key and rc == 0:
                    state["stdout_memory"].add(stdout_key)
                    if stdout_norm_key:
                        state["stdout_norm_memory"].add(stdout_norm_key)

                # ── Calcula reward de RL ───────────────────────────────
                if is_duplicate:
                    # Duplicata exata: ZERO reward, sem gradient.
                    # (gradient com advantage negativo aqui causaria
                    #  o "choque" que desorientava o modelo)
                    rl_reward = 0.0
                    log(f"⚠️  Gen {gen}: duplicata exata — update ignorado")
                elif not stdout_is_new and r_val >= 0.5:
                    # Stdout já visto (ex: "hello world" de outra forma):
                    # reward zero para desincentivar variações sem criatividade
                    rl_reward = 0.0
                    log(f"⚠️  Gen {gen}: stdout repetido ({repr(stdout_key[:25])}) — sem reward RL")
                else:
                    rl_reward = r_val

                # ── Melhor resultado ──────────────────────────────────
                if r_val >= state["best_reward"] or not state["best_code"]:
                    state["best_reward"] = r_val
                    state["best_code"]   = code
                    state["best_gen"]    = gen
                    log(f"🏆 Gen {gen}: novo recorde! reward={r_val:.2f}  →  {repr(code[:50])}")
                    # Salva checkpoint do melhor modelo
                    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_BEST, pretrain_opt=pretrain_opt)
                    log(f"💾 Best checkpoint salvo (gen {gen})")

                # ── Premium: nova habilidade distinta? ────────────────
                if rc == 0 and stdout and stdout_is_new:
                    saved = save_premium_checkpoint(
                        model, rl_opt, sup_opt, pretrain_opt,
                        state, code, stdout, r_val, gen,
                        premium_stdout_set=premium_stdout_set,
                    )
                    if saved:
                        state["premium_count"] += 1
                        log(f"🌟 Gen {gen}: habilidade premium salva! "
                            f"reward={r_val:.2f}  stdout={repr(stdout[:30])}  "
                            f"total={state['premium_count']}")

                # ── REINFORCE ─────────────────────────────────────────
                # IMPORTANTE: duplicata exata → pular update completamente.
                # Rodar REINFORCE com advantage = 0 - baseline < 0 causava
                # um gradiente grande que desorientava o modelo ("choque").
                baseline = (sum(list(state["reward_hist"])[-20:]) /
                            max(1, min(20, len(state["reward_hist"]))))
                if not is_duplicate:
                    _, ent_val = reinforce_update(
                        model, rl_opt, log_probs, entropies, rl_reward, baseline
                    )
                    if entropies is not None:
                        state["last_entropy"] = ent_val
                else:
                    # Duplicata: só atualiza display de entropia, sem gradiente
                    if entropies is not None:
                        state["last_entropy"] = entropies.mean().item()

                # ── Treino supervisionado em sucesso ──────────────────
                # Só reforça se o stdout é genuinamente novo — impede
                # sobretreinar em "hello world" e suas variações.
                if rc == 0 and stdout and r_val >= 0.5 and not is_duplicate and stdout_is_new:
                    supervised_on_success(model, sup_opt, code, steps=15)
                    if r_val >= 1.0:
                        log(f"✨ Gen {gen}: funcionou! reward={r_val:.2f}  stdout={repr(stdout[:30])}")

                # ── Temperatura ───────────────────────────────────────
                state["temp"] = max(TEMP_MIN, state["temp"] * TEMP_DECAY)
                if (len(state["recent_codes"]) >= 15
                        and state["diversity"] < 0.30
                        and gen % 10 == 0):
                    old_temp = state["temp"]
                    state["temp"]        = min(TEMP_START, state["temp"] * 3.0)
                    state["temp_resets"] += 1
                    log(f"🔄 Gen {gen}: diversidade={state['diversity']:.0%} → "
                        f"temp reset {old_temp:.3f}→{state['temp']:.3f}")

                # ── Detecção de entropia travada → pré-treino automático ──
                # Condição: entropia alta (geração aleatória) E reward médio baixo
                # por N gerações seguidas → modelo preso em equilíbrio ruim.
                # Ex clássico: gerar '# [lixo]' valia 0.25, ficava preso nisso.
                # Com o fix do exploit esse reward caiu para ~0.01, mas a
                # detecção serve de salvaguarda para outros equilibrios ruins.
                cur_avg = (sum(list(state["reward_hist"])[-20:]) /
                           max(1, min(20, len(state["reward_hist"]))))
                if (state["last_entropy"] >= ENTROPY_STUCK_THRESHOLD
                        and cur_avg < ENTROPY_STUCK_REWARD):
                    state["entropy_stuck_count"] += 1
                else:
                    state["entropy_stuck_count"] = 0   # reset se saiu do buraco

                if state["entropy_stuck_count"] >= ENTROPY_STUCK_PATIENCE:
                    state["entropy_stuck_count"] = 0
                    log(f"🆘 Gen {gen}: entropia={state['last_entropy']:.2f} travada  "
                        f"avg={cur_avg:.2f} por {ENTROPY_STUCK_PATIENCE} gens → "
                        f"pré-treino automático de re-ancoragem ({ENTROPY_STUCK_PT_STEPS} steps)!")
                    state["phase"]        = "pretrain"
                    state["current_code"] = f"[re-ancoragem automática] 0/{ENTROPY_STUCK_PT_STEPS}"
                    refresh(force=True)

                    def _auto_pt_cb(step, total, loss):
                        state["current_code"] = (
                            f"[re-ancoragem automática] step {step}/{total}  loss={loss:.4f}")
                        refresh()
                        if step > 0 and step % 50 == 0:
                            save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)

                    pretrain(model, ENTROPY_STUCK_PT_STEPS, PRETRAIN_LR_CONT,
                             _auto_pt_cb, opt=pretrain_opt)
                    # Reseta temperatura para explorar após re-ancoragem
                    state["temp"]         = TEMP_START
                    state["phase"]        = "evolve"
                    state["current_code"] = ""
                    log(f"✅ Re-ancoragem concluída. Temp resetada para {TEMP_START}")
                    refresh(force=True)

                # ── Checkpoint periódico ───────────────────────────────
                if gen % CHECKPOINT_EVERY == 0:
                    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
                    state["last_ckpt_gen"] = gen
                    log(f"💾 Checkpoint salvo (gen {gen})")

                # ══ MUTAÇÃO DE ARQUITETURA ════════════════════════════
                if gen > 0 and gen % MUTATION_EVERY == 0:
                    state["phase"] = "mutation"
                    log(f"🧬 Gen {gen}: avaliando mutações...")
                    refresh(force=True)

                    current_cfg       = state["config"]
                    # FIX: a referência tem que ser o loss do modelo REAL e já
                    # treinado (model), não de um modelo novo do zero. Usar
                    # eval_config() aqui criava um modelo com pesos aleatórios
                    # e só 80 steps de treino — o loss dele é sempre muito pior
                    # que o do modelo atual, então QUALQUER mutação parecia
                    # "melhora" e acabava substituindo um modelo bom por um
                    # pior. Isso é o motivo da mutação estar frágil/quebrando
                    # o modelo.
                    ref_loss          = _eval_model_loss(model)
                    best_mut_loss     = ref_loss
                    best_mut_cfg      = None
                    best_mut_m        = None

                    for t in range(MUTATION_TRIALS):
                        mut_cfg = mutate_config(current_cfg)
                        state["current_code"] = (
                            f"[mutação {t+1}/{MUTATION_TRIALS}] "
                            f"embed={mut_cfg['embed_dim']} layers={mut_cfg['n_layers']} "
                            f"heads={mut_cfg['n_heads']}"
                            + (f" experts={mut_cfg.get('n_experts')}" if mut_cfg.get("use_moe") else "")
                        )
                        refresh()
                        try:
                            mut_m, mut_loss = eval_mutation_from_model(model, mut_cfg)
                        except Exception as e:
                            # Antes: falha silenciosa (except: continue), então
                            # uma mutação inválida (ex: shape incompatível,
                            # assert de embed_dim % n_heads) sumia sem deixar
                            # rastro nenhum no log. Agora fica visível.
                            log(f"⚠️  Mutação {t+1}/{MUTATION_TRIALS} falhou: "
                                f"{type(e).__name__}: {e}")
                            continue
                        if mut_loss < best_mut_loss:
                            best_mut_loss = mut_loss
                            best_mut_cfg  = mut_cfg
                            best_mut_m    = mut_m

                    if best_mut_cfg is not None:
                        model           = best_mut_m
                        state["config"] = best_mut_cfg
                        state["n_params"] = model.n_params
                        rl_opt  = torch.optim.AdamW(model.parameters(), lr=REINFORCE_LR)
                        sup_opt = torch.optim.AdamW(model.parameters(), lr=SUPERVISED_LR)
                        pretrain_opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR,
                                                          weight_decay=1e-4)
                        desc = (f"embed={best_mut_cfg['embed_dim']} "
                                f"L={best_mut_cfg['n_layers']} H={best_mut_cfg['n_heads']}"
                                + (f" E={best_mut_cfg.get('n_experts')}" if best_mut_cfg.get("use_moe") else ""))
                        state["mutation_log"].append(f"✅ {desc} (gen {gen})")
                        log(f"🧬 Mutação aceita! {desc}  loss {ref_loss:.4f}→{best_mut_loss:.4f}")
                        # Salva checkpoint após mutação aceita
                        save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)
                        state["last_ckpt_gen"] = gen
                        log(f"💾 Checkpoint pós-mutação salvo")
                    else:
                        state["mutation_log"].append(f"❌ manteve (gen {gen})")
                        log(f"🧬 Sem melhora na mutação (ref_loss={ref_loss:.4f})")

                    state["phase"]        = "evolve"
                    state["current_code"] = ""

                # ── Comando de teclado (tecla P = novo pré-treino) ────
                try:
                    cmd = _cmd_queue.get_nowait()
                    parts = cmd.strip().upper().split()
                    if parts and parts[0] == "P":
                        pt_steps = PRETRAIN_STEPS
                        if len(parts) > 1:
                            try:
                                pt_steps = int(parts[1])
                            except ValueError:
                                pass
                        log(f"⌨️  Pré-treino manual solicitado ({pt_steps} steps)...")
                        state["phase"]        = "pretrain"
                        state["current_code"] = f"[pré-treino manual] 0/{pt_steps}"
                        refresh()

                        def _manual_cb(step, total, loss):
                            state["current_code"] = (
                                f"[pré-treino manual] step {step}/{total}  loss={loss:.4f}")
                            refresh()
                            if step > 0 and step % 50 == 0:
                                save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)

                        pretrain(model, pt_steps, PRETRAIN_LR_CONT, _manual_cb,
                                 opt=pretrain_opt)

                        state["phase"]        = "evolve"
                        state["current_code"] = ""
                        log(f"✅ Pré-treino manual concluído ({pt_steps} steps)")
                        refresh()
                except queue.Empty:
                    pass

                refresh(force=True)

        except KeyboardInterrupt:
            pass

    # ── Salva checkpoint final ao sair ────────────────────────────────
    console.print("\n[yellow]💾 Salvando checkpoint final...[/yellow]")
    save_checkpoint(model, rl_opt, sup_opt, state, CHECKPOINT_LAST, pretrain_opt=pretrain_opt)

    # ── Resumo final ──────────────────────────────────────────────────
    console.print("\n[bold blue]══ RESUMO FINAL ══[/bold blue]")
    console.print(f"   Gerações: {state['gen']}")
    console.print(f"   Melhor reward: {state['best_reward']:.2f}")
    console.print(f"   Melhor código (gen {state['best_gen']}):")
    console.print(f"   [bold cyan]{repr(state['best_code'])}[/bold cyan]")
    console.print(f"   Arquitetura final: {state['config']}")
    console.print(f"   Parâmetros: {state['n_params']:,}")
    console.print(f"   Checkpoints em: [bold]{os.path.abspath(CHECKPOINT_DIR)}[/bold]")
    console.print(f"   Habilidades premium: [bold magenta]{state['premium_count']}[/bold magenta] salvas em: [bold]{os.path.abspath(PREMIUM_DIR)}[/bold]\n")
    console.print("[bold green]🧬 A IA evoluiu. Até a próxima![/bold green]\n")


if __name__ == "__main__":
    main()