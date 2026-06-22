/* Inference for Llama-2 Transformer model in pure C
 * With added CUDA support initially drawing from
 * https://github.com/ankan-ban/llama2.cu/blob/master/llama2.cu
 * and structured in a way that hopefully makes keeping it
 * up-to-date straightforward.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>

// Each CUDA function call should be checked for errors.
#define CUCHK(err) cuda_check((err), __FILE__, __LINE__)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

cublasHandle_t g_cublas_handle = nullptr;

void create_cublas_handle() {
    cublasStatus_t stat = cublasCreate(&g_cublas_handle);  // FIXME cublasDestroy
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
}
void destroy_cublas_handle() {
    cublasStatus_t stat = cublasDestroy(g_cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
}
#endif

// ----------------------------------------------------------------------------
// Transformer model

// ============================================================================
// Config: 模型超参数,即 .bin 文件最开头的 7 个 int(共 28 字节)的"蓝图"。
// 注释中的示例数值来自 stories15M.bin。
// ============================================================================
typedef struct {
    int dim;         // 主干隐藏维度 (hidden size)。x 向量的长度。  例: 288
    int hidden_dim;  // FFN 中间层维度 (升维后的宽度)。           例: 768
    int n_layers;    // 解码层数量 (decoder layer 个数)。          例: 6
    int n_heads;     // Query 注意力头数。 head_size = dim/n_heads. 例: 6 (→head_size=48)
    int n_kv_heads;  // Key/Value 头数。 < n_heads 即 GQA/MQA 共享. 例: 6 (=n_heads,即普通MHA)
    int vocab_size;  // 词表大小。 文件里若为负=不共享 lm_head 权重. 例: 32000
    int seq_len;     // 最大序列长度 (KV cache / 位置上限)。        例: 256
} Config;

// CUDA NOTE: The TransformerWeights structure will be stored on the host, 
// but all of the pointers in the structure will point to data on the GPU.
// The checkpoint file is mmap-ed to the host and the weights portion 
// is allocated on and copied to the GPU.  Then, memory_map_weights() updates  
// these structure pointers to point to the proper location.  Happily, this
// function is the same for both C and CUDA.
typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

// CUDA NOTE: The RunState structure will be stored on the host, but all of the
// pointers in the structure will point to data on the GPU, created via
// cudaMalloc.  The exception is logits which is the final result of the
// transformer & is copied from the GPU as the last step in the transformer
// and is used by the host.
typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
#ifdef USE_CUDA
    float *logits_gpu; // output logits in GPU
#endif
    float *logits; // output logits in CPU
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

#ifdef USE_CUDA
void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CUCHK(cudaMalloc((void**)&s->x, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->xb, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->xb2, p->dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->q, p->dim * sizeof(float)));
    // ---- KV Cache:显存里缓存所有历史 token 的 K/V,供注意力复用(启动时一次性开满)----
    //   大小 = n_layers × seq_len × kv_dim × sizeof(float),K 和 V 各一份。
    //   stories15M:6 × 256 × 288 × 4B = 1,769,472B ≈ 1.69 MiB 一份,K+V ≈ 3.4 MiB。
    //
    //   内存布局 [layer][seq_len][kv_dim](一维连续铺开):
    //
    //     key_cache (V 同结构):
    //     ┌──────────── 层0 ────────────┬──── 层1 ────┬ ... ┬──── 层5 ────┐
    //     │ pos0  pos1  pos2 ... pos255 │ pos0 ...    │     │ pos0 ...    │
    //     │ [288] [288] [288]...[288]   │             │     │             │
    //     └─────────────────────────────┴─────────────┴─────┴─────────────┘
    //       │←─ seq_len × kv_dim = 256×288 = 73728 ─→│   每层这么大,共 6 层
    //       每个 pos 槽位存该位置的 K 向量(kv_dim=288 个 float)
    //
    //       定位"第 l 层、第 pos 个位置": loff + pos*kv_dim,  loff = l*seq_len*kv_dim
    //       (见 forward 里 s->k = key_cache + loff + pos*kv_dim;算完 K 直接落进该槽位)
    //
    //   • 按 seq_len(最大长度)预留:即使只生成几个 token 也开满 256 个位置(避免运行时反复申请)。
    //   • 用 kv_dim 而非 dim:本模型 MHA 时 kv_dim=dim=288;GQA 时 kv_dim 更小 → KV Cache 显著缩小。
    //   ⚠ 大模型会爆炸:Llama-2-7B(32层,seq=4096,kv_dim=4096)单份 ≈2GiB,K+V ≈4GiB
    //     → 这就是长上下文吃显存的根源,以及 GQA / KV 量化 / PagedAttention 等优化的意义。
    //
    //   随生成逐步填充(prefill 填完 prompt,decode 每步往后写 1 个 pos):
    //     已写: [pos0][pos1][pos2] ... [当前pos] │ 预留未用 ... [pos255]
    //            └── 注意力读这一段(0..pos)──┘ └── 还没写到的空槽 ──┘
    CUCHK(cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->att, p->n_heads * p->seq_len * sizeof(float)));
    CUCHK(cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(float)));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}
#else
void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float *)calloc(p->dim, sizeof(float));
    s->xb = (float *)calloc(p->dim, sizeof(float));
    s->xb2 = (float *)calloc(p->dim, sizeof(float));
    s->hb = (float *)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float *)calloc(p->hidden_dim, sizeof(float));
    s->q = (float *)calloc(p->dim, sizeof(float));
    s->key_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float *)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float *)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef USE_CUDA
void free_run_state(RunState* s) {
    CUCHK(cudaFree(s->x));
    CUCHK(cudaFree(s->xb));
    CUCHK(cudaFree(s->xb2));
    CUCHK(cudaFree(s->hb));
    CUCHK(cudaFree(s->hb2));
    CUCHK(cudaFree(s->q));
    CUCHK(cudaFree(s->att));
    CUCHK(cudaFree(s->logits_gpu));
    free(s->logits);
    CUCHK(cudaFree(s->key_cache));
    CUCHK(cudaFree(s->value_cache));
}
#else
void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}
#endif

// memory_map_weights: 权重区在文件里是一段连续的 float 流(无名字、无分隔)。
// 这里按"固定顺序"把一根大指针 ptr 切成各个权重矩阵——每赋一个指针,就把 ptr
// 往后推进该矩阵的元素个数。顺序必须和 export.py 写出来的顺序完全一致。
void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    float* base = ptr; // [观测] 记下权重区起点,用于计算每个矩阵的偏移

    w->token_embedding_table = ptr;                              // (vocab, dim)  词嵌入表
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;                                     // (layer, dim)  注意力前 RMSNorm 增益
    ptr += n_layers * p->dim;
    w->wq = ptr;                                                 // (layer, dim, n_heads*head_size)  Q 投影
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;                                                 // (layer, dim, n_kv_heads*head_size) K 投影
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;                                                 // (layer, dim, n_kv_heads*head_size) V 投影
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;                                                 // (layer, dim, dim)  注意力输出投影
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;                                     // (layer, dim)  FFN 前 RMSNorm 增益
    ptr += n_layers * p->dim;
    w->w1 = ptr;                                                 // (layer, hidden, dim)  FFN 门控升维
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;                                                 // (layer, dim, hidden)  FFN 降维
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;                                                 // (layer, hidden, dim)  FFN 数据升维
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;                                   // (dim,)  最终 RMSNorm 增益
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE) —— 老格式遗留,跳过
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE) —— RoPE 现在运行时算
    // 分类头:若共享则直接复用词嵌入表(省一份大矩阵),否则用紧跟其后的权重
    w->wcls = shared_weights ? w->token_embedding_table : ptr;

    // ---- [观测] 打印每个权重矩阵的形状 / 元素数 / 在权重区内的偏移 ----------
    unsigned long long L = n_layers, D = p->dim, H = p->hidden_dim;
    unsigned long long Qd = p->n_heads * head_size, KVd = p->n_kv_heads * head_size;
    fprintf(stderr, "\n========================= [memory_map_weights] 权重布局 =========================\n");
    fprintf(stderr, "  分层列: 'L×单层' 表示该块内按 %llu 层首尾相接,每层独立权重(w->X + l*单层)\n", L);
    fprintf(stderr, "  %-18s %-22s %13s  %4s  %11s  %11s\n",
            "名称", "形状", "总元素", "分层", "单层元素", "偏移(elem)");
    fprintf(stderr, "  ------------------------------------------------------------------------------\n");
    // layered=1 的块:总元素 = L × per_layer,块内按 6 层细分;layered=0:整块,全模型 1 份
    #define WLOG(field, shapestr, layered, per_layer) do { \
        unsigned long long _pl = (unsigned long long)(per_layer); \
        unsigned long long _tot = (layered) ? (L * _pl) : _pl; \
        if (layered) \
            fprintf(stderr, "  %-18s %-22s %13llu  %4s  %11llu  %11lld\n", #field, shapestr, \
                    _tot, "L×", _pl, (long long)(w->field - base)); \
        else \
            fprintf(stderr, "  %-18s %-22s %13llu  %4s  %11s  %11lld\n", #field, shapestr, \
                    _tot, "—", "(整块)", (long long)(w->field - base)); \
    } while(0)
    WLOG(token_embedding_table, "(vocab, dim)",        0, (unsigned long long)p->vocab_size * D);
    WLOG(rms_att_weight,        "(layer, dim)",         1, D);
    WLOG(wq,                    "(layer, dim, Qd)",     1, D * Qd);
    WLOG(wk,                    "(layer, dim, KVd)",    1, D * KVd);
    WLOG(wv,                    "(layer, dim, KVd)",    1, D * KVd);
    WLOG(wo,                    "(layer, Qd, dim)",     1, Qd * D);
    WLOG(rms_ffn_weight,        "(layer, dim)",         1, D);
    WLOG(w1,                    "(layer, hidden, dim)", 1, D * H);
    WLOG(w2,                    "(layer, dim, hidden)", 1, H * D);
    WLOG(w3,                    "(layer, hidden, dim)", 1, D * H);
    WLOG(rms_final_weight,      "(dim,)",               0, D);
    WLOG(wcls,                  "(vocab, dim)",         0, (unsigned long long)p->vocab_size * D);
    #undef WLOG
    fprintf(stderr, "  ------------------------------------------------------------------------------\n");
    fprintf(stderr, "  注: Qd=n_heads*head_size=%llu, KVd=n_kv_heads*head_size=%llu\n", Qd, KVd);
    fprintf(stderr, "      分层块 'L×' 内含 %llu 个独立子矩阵(各层 W_Q/W_K/W_V/... 互不共享)%s\n",
            L, shared_weights ? "" : "");
    if (shared_weights)
        fprintf(stderr, "      wcls 与 token_embedding 共享(偏移相同 0,文件不单独存储)\n");
    fprintf(stderr, "================================================================================\n\n");
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    // .bin 开头就是一个 Config 结构(7 个 int)。fread 一次性把"蓝图"读进来。
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    // vocab_size 为负 = lm_head(分类头)不与 token embedding 共享权重(这是个 hack 约定)。
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    // ---- [观测] 打印解析出来的 Config 超参 ----------------------------------
    int head_size_dbg = config->dim / config->n_heads;
    fprintf(stderr, "\n==================== [read_checkpoint] Config ====================\n");
    fprintf(stderr, "  dim            = %d   (主干隐藏维度)\n", config->dim);
    fprintf(stderr, "  hidden_dim     = %d   (FFN 中间层维度)\n", config->hidden_dim);
    fprintf(stderr, "  n_layers       = %d   (解码层数)\n", config->n_layers);
    fprintf(stderr, "  n_heads        = %d   (Query 头数)\n", config->n_heads);
    fprintf(stderr, "  n_kv_heads     = %d   (KV 头数, < n_heads 即 GQA)\n", config->n_kv_heads);
    fprintf(stderr, "  head_size      = %d   (= dim / n_heads)\n", head_size_dbg);
    fprintf(stderr, "  vocab_size     = %d\n", config->vocab_size);
    fprintf(stderr, "  seq_len        = %d   (最大序列长度)\n", config->seq_len);
    fprintf(stderr, "  shared_weights = %d   (lm_head 是否复用 embedding)\n", shared_weights);
    fprintf(stderr, "  file_size      = %.2f MB\n", (double)(*file_size) / (1024.0 * 1024.0));
    fprintf(stderr, "==================================================================\n");
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
#ifdef USE_CUDA
    // allocate & copy mmap data to the gpu first
    // TODO: allocate & copy just a portion to the GPU if the weights are too big
    // to fit in the GPU, then copy the data only as needed while running.
    float* weights_ptr;
    size_t weights_size = *file_size - sizeof(Config);
    CUCHK(cudaMalloc((void**)&weights_ptr, weights_size));
    CUCHK(cudaMemcpy(weights_ptr, *data + sizeof(Config)/sizeof(float), weights_size, cudaMemcpyHostToDevice));
#else
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
#endif
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
#ifdef USE_CUDA
    // we cudaMalloc a region of memory, then hand the address to
    // the token_embedding_table field.  Free it here.
    CUCHK(cudaFree(t->weights.token_embedding_table));
#endif
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

#ifdef USE_CUDA
// Utility routine to divide a into ceiling of b parts
int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

const int num_threads_lrg = 1024;
const int num_threads_med = 256;

// ============================================================================
// RMSNorm(Root Mean Square Normalization,均方根归一化)
//
// ★ 一句话直觉:把向量"调到统一音量,但不改旋律",再逐维微调。
//   目的是让每层拿到的数值都在稳定范围内,不会越传越大而失控。
//
// ─────────────────────── 用具体数字走一遍(假设只有4维)───────────────────────
//   输入 x = [3, -4, 2, 1]
//   ① 算"整体音量" RMS:每个数平方→求平均→开根号
//        平方: [9, 16, 4, 1] → 和=30 → /4=7.5 → √7.5 ≈ 2.74  (这就是 RMS)
//   ② 每个数都 ÷ 2.74(除的是同一个数!所以形状/正负/相对大小都不变):
//        [3,-4,2,1] / 2.74 = [1.10, -1.46, 0.73, 0.37]
//   ③ 再 × weight(每维一个,训练学出来的"音量微调"):
//        归一化 [1.10,-1.46,0.73,0.37] ⊙ weight [1.0,2.0,0.5,1.5]
//        = [1.10, -2.92, 0.37, 0.55]  ← 输出 o
//
//   归一化前后对比(柱子相对高低/正负完全不变,只是整体缩到标准大小):
//
//     归一化前 x:                          归一化后 (÷2.74):
//      3 │ ██▌                              1.10 │ ██▌
//      2 │ █▌            ── ÷ 2.74 ──►       0.73 │ █▌
//      1 │ ▌                                0.37 │ ▌
//      0 ┼────────                          0.00 ┼────────
//     -4 │███▌                             -1.46 │███▌
//          x0 x1 x2 x3                             x0 x1 x2 x3
//        (整体偏大、起伏大)                  (拉回标准音量,但起伏比例一模一样)
//
//   类比:每层是接力选手,向量越传可能越"吵"。RMSNorm 是每层入口的"音量旋钮",
//        不管进来多大声,先统一调到标准音量再往下算 → 数值稳定。
//
// ─────────────────────── 输入 / 输出(都是长度 dim=288 的【向量】)──────────────
//   x[288]      ← 输入:当前隐藏状态(一个 token 的向量)
//   weight[288] ← 每维的可学习增益(见下"weight 说明")
//   o[288]      → 输出:归一化并按 weight 缩放后的向量
//
//   公式:o[i] = (x[i] / rms) · weight[i],  其中 rms = sqrt(mean(x²)+ε)
//   注意:rms 对【整个向量】只算一次(标量),广播给每维;weight 是逐维相乘。
//
//   ★ "× weight" 是【逐元素相乘 ⊙】(Hadamard),不是点乘、不是矩阵乘!
//     即 o[i] = 归一化值[i] · weight[i],下标一一对应,各维独立,不跨维求和。
//        归一化 [a0, a1, ..., a287]
//                ×    ×         ×        ← 对位相乘
//        weight [g0, g1, ..., g287]
//        输出 o [a0·g0, a1·g1, ..., a287·g287]   ← 仍是 288 维【向量】
//     判别窍门:输出还是 288 维 → 逐元素乘;若变成 1 个标量 → 那才是点乘(这里不是)。
//     原因:weight 的意义是"给每一维单独配一个增益",所以必须对位、各维独立。
//   没有矩阵乘:RMSNorm = "一次全局归约(求rms)+ 逐元素缩放",不涉及矩阵乘法。
//
// ─────────────────────── weight 说明(常被问到)──────────────────────────────
//   • 它是 RMSNorm 里唯一可学习的参数(归一化那步是纯计算、无参数)。
//   • 形状 = 长度 dim=288 的【一维向量】(不是矩阵!),值多在 1.0 附近。
//     注意:长度是 dim(288),不是 vocab(32000)!vocab 只跟 embedding/lm_head 有关。
//   • 文件里有三个、且【每层各一套】(rms_final 除外):
//       rms_att_weight   每层1个(注意力前norm)  → 6层共 6×288=1728
//       rms_ffn_weight   每层1个(FFN前norm)     → 6层共 1728
//       rms_final_weight 全模型仅1个(收尾norm)  → 288
//     取第 l 层:w->rms_att_weight + l*dim。HF 里对应 input_layernorm.weight 等。
//
//   • 全部 RMSNorm weight 总元素数:
//       = dim × (n_layers×2 + 1)            ← 每层2个(att前/ffn前) + 1个最终收尾
//       = 288 × (6×2 + 1) = 288 × 13 = 3744 个 float
//     ★ 易错点:基数是 dim(288),不是 vocab;那个"+1"是【最终收尾的独立 norm】
//       (6层全算完后、进 lm_head 前做一次),不是"被6层共享"。各层的 att/ffn norm 也互不共享。
//
//       x ─层0─►层1─►...─►层5─►[rms_final 归一化一次]─► lm_head
//          每层内部各有自己的            ↑ 收尾,全模型仅此一次(非共享)
//          rms_att + rms_ffn(各层独立)
//
// ─────────────────────── 为什么"一进层就 norm"(Pre-Norm)─────────────────────
//   Llama 用 Pre-Norm:先 norm 再做 QKV/FFN(norm 是变换前的"预处理",不是收尾)。
//   QKV 投影吃的是归一化后的 xb,不是原始 x。且残差捷径用的是【原始 x】,所以
//   norm 输出写到 xb、x 保持不变:  x ──norm──► xb ──变换──► 结果;  x + 结果 = 新x
//   (2017 原始 Transformer 是 Post-Norm:先变换后 norm;现代大模型多改用 Pre-Norm,更稳)
//
// ─────────────────────── GPU 实现(三阶段)──────────────────────────────────
//   <<<1, 1024>>> 一个 block、1024 线程协作处理这一个向量:
//   x: [x0 x1 ... x287]
//        │① 平方求和(1024线程各算部分和 → cub::BlockReduce 合成1个总和)
//        ▼
//      Σx² ─/n─► 均方 ─+ε, 1/sqrt─► ss(②由0号线程算出的标量,广播给全部线程)
//        │③ 逐元素缩放
//        ▼
//   o: [g0·ss·x0 , g1·ss·x1 , ... , g287·ss·x287]   (g=weight)
// ============================================================================
__global__ void rmsnorm_kernel(float* o, float* x, float* weight, int size, int elementsPerThread) {
    // —— 阶段①:归约求平方和 —— 每个线程先累加自己负责那几个元素的 x[j]²
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_lrg;   // 线程 t 负责 t, t+1024, t+2048...
        if (j < size)
            ss += x[j] * x[j];
    }
    // ---- 用 NVIDIA CUB 库做"块内归约":把 1024 个线程各自的部分和合并成 1 个总和 ----
    //   ① cub::BlockReduce<float, 1024>:模板参数 = (归约数据类型, block 内线程数,
    //      须与 blockDim 一致)。只是定义类型,还没计算。
    //   ② TempStorage temp:CUB 归约所需的共享内存"工作台"(线程间交换部分结果用),
    //      __shared__ 表示整个 block 共享;大小由 CUB 按上面的模板参数算好。
    //   ③ .Sum(ss):传入【每个线程自己】的部分和 ss,CUB 用树形并行规约(log2(1024)=10步,
    //      两两相加逐层折半)把 1024 个 ss 全加起来。返回的总和只有 0 号线程有效。
    //
    //      t0:x0²  t1:x1² ... t287:x287²  t288..1023:0
    //          └────┬────┴─────┬──────────┘
    //               ▼ BlockReduce(...).Sum(ss)  (树形规约)
    //          ss(t0)=Σx² 总平方和(仅 0 号线程拿到 → 下面阶段②用它)
    //   自己手写块内求和要操心同步/共享内存/warp shuffle/bank冲突,CUB 封装好且高度优化。
    using BlockReduce = cub::BlockReduce<float, num_threads_lrg>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    // —— 阶段②:把阶段①的【平方和】加工成最终缩放因子 ss = 1/rms,放进共享内存广播 ——
    //   进来时 ss = 平方和(Σx²)。以 x=[3,-4,2,1] 为例,ss=9+16+4+1=30:
    //     ① ss /= size       : 30/4 = 7.5        → 均方(mean of squares)
    //     ② ss += 1e-5f       : 7.5+ε ≈ 7.5       → 加 ε 防止除零/极端值
    //     ③ ss = 1/sqrtf(ss) : 1/√7.5 ≈ 0.365    → 1/rms,这才是最终缩放因子
    //   ★ 算的是【倒数 1/rms】而非 rms:因为归一化要"÷rms",而 GPU 上除法比乘法慢,
    //     先求一次倒数,阶段③就能用快速的乘法 o[i]=x[i]*(1/rms) 代替慢除法。
    //   ★ 只让 0 号线程算:ss 是整个向量共享的一个标量,1024 线程算出来都一样,
    //     让一个线程算一次、写进 shared_ss 广播,避免 1024 倍重复计算。
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;            // ① 平方和 → 均方(/ n)
        ss += 1e-5f;           // ② 加 ε 防止除零
        ss = 1.0f / sqrtf(ss); // ③ 取 1/sqrt → ss = 1/rms(例:30→7.5→0.365)
        shared_ss = ss;        // ④ 写入共享内存,准备广播给其余线程
    }
    __syncthreads();           // 屏障:等 0 号线程算完写好,其余线程才能读
    ss = shared_ss;            // 所有线程拿到同一个标量 ss(=1/rms,例 0.365)

    // —— 阶段③:逐元素归一化+缩放 —— o[i] = weight[i] · (ss · x[i])
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_lrg;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}
void rmsnorm(float* o, float* x, float* weight, int size) {
    int elementsPerThread = divUp(size, num_threads_lrg);
    rmsnorm_kernel <<<1, num_threads_lrg >>> (o, x, weight, size, elementsPerThread);
}
#else
void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
#endif

#ifdef USE_CUDA
// ============================================================================
// softmax_gpu:把一组分数变成"和为1的权重"(块内 1024 线程协作,原地覆盖 x[0..size))
//
// 【目的】注意力打分 att=[1.2, 3.5, 0.8, 2.0] 是无范围的原始分,softmax 把它压成
//   概率分布 [0.08, 0.65, 0.05, 0.22](全正、和=1)→ 当作"该给谁多少注意力"的权重。
//
// 【公式】softmax(x)[i] = e^(x[i]) / Σ e^(x[j])
//   但直接 e^x 在 x 大时会溢出,所以先减去最大值(数值稳定,结果不变):
//             = e^(x[i] - max) / Σ e^(x[j] - max)
//
// 【三段,每段都是"线程分摊 + CUB 块内归约"】(size = pos+1 个分数):
//
//   段1 求最大值 max(为了数值稳定):
//       线程各扫自己负责的元素求局部max → cub::BlockReduce(Max) 合成全局 max
//       x=[1.2, 3.5, 0.8, 2.0] ───► max = 3.5
//
//   段2 减max、取指数、求和:
//       x[i] = exp(x[i]-max);  各线程累加局部和 → cub::BlockReduce(Sum) 合成总和
//       exp: [e^-2.3, e^0, e^-2.7, e^-1.5] = [0.10, 1.0, 0.067, 0.22]  → sum=1.39
//
//   段3 归一化:每个元素 ÷ sum
//       [0.10,1.0,0.067,0.22]/1.39 = [0.072, 0.72, 0.048, 0.16]  ← 和=1,即权重
//
//   每段之间靠 __syncthreads + shared_val 把归约结果(max/sum)广播给所有线程。
//
//   ── 一图总览(att=[1.2,3.5,0.8,2.0])──────────────────────────────────────
//     原始分  [1.2,  3.5,  0.8,  2.0]
//                │ 段1: 求 max=3.5                 (CUB Max 归约 + 广播)
//                │ 段2: 减max→exp→[0.10,1.0,0.067,0.22]
//                │        求 sum=1.39               (CUB Sum 归约 + 广播)
//                │ 段3: 每个 ÷1.39
//                ▼
//     权重    [0.072, 0.72, 0.048, 0.16]  和=1  ← cat(原始分最高)拿到 72% 注意力
//   ──────────────────────────────────────────────────────────────────────────
//   直觉:exp 放大差距 → 原始分越高,权重越大 → 注意力自动偏向最相关的历史 token。
//
// 【为何要"协作"而非各算各的】softmax 需要全局信息(整行 max、整行 sum),所以每段都是:
//   线程各算局部值 → CUB BlockReduce 合成全局 → 0号线程广播 → 全员用它继续。
//   (对比打分那步是"分摊型":线程各写各的 att[t],互不通信。)
//
// 【线程怎么分摊】网格跨步:线程 tid 负责 x[tid], x[tid+1024], ...(本例 size 小,
//   基本是一线程一个元素;att 最长 256<1024,所以大多一人一个,多余线程不参与)。
// ============================================================================
__device__ void softmax_gpu(float* __restrict__ x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;          // 步长=线程数,做网格跨步

    // —— 段1:求最大值 max(数值稳定用)——
    // 每个线程先求自己负责那些元素的局部最大
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_lrg>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());  // CUB 归约:1024个局部max → 1个全局max
    if (threadIdx.x == 0) {        // 0号线程拿到结果,放共享内存广播
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;          // 所有线程拿到同一个 max

    // —— 段2:每个元素减max、取exp,同时累加得到总和 sum ——
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);   // 原地:分数 → exp 值
        sum += x[i];                    // 累加局部和
    }
    sum = BlockReduce(temp).Sum(sum);   // CUB 归约:1024个局部和 → 总和
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;              // 所有线程拿到同一个 sum

    // —— 段3:归一化,每个 exp 值 ÷ sum → 得到权重(和为1)——
    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}
#endif
void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ============================================================================
// matmul:矩阵 × 向量(线性层/投影)。这是 Transformer 里真正的"矩阵乘",和 RMSNorm
//   的"逐元素"完全不同 —— 输出的每一维要对输入【整个向量】加权求和。
//
//   参数 n / d 含义:n = 输入维度(x 的长度),d = 输出维度(xout 的长度)。
//     W 是 d×n 矩阵,把 n 维输入投影成 d 维输出。记忆:n=iNput 在前,d=输出在后。
//     例:QKV 投影 n=d=288;FFN 升维 n=288,d=768;FFN 降维 n=768,d=288;
//         lm_head n=288,d=32000。可见 d 与 n 常不相等。
//
//   计算:xout[d] = W[d×n] · x[n]      (W 是权重矩阵,x 是输入向量,xout 是输出向量)
//
//   输出第 i 维 = W 的第 i 行 与 x 做点乘(逐元素乘再求和):
//       xout[i] = Σ_j  W[i][j] · x[j]      (j 从 0 到 n-1)
//
//        W (d 行 × n 列)        x (n)        xout (d)
//       ┌──────────────┐       ┌──┐         ┌──┐
//   行0 │ w00 w01 .. w0,n-1│·  │x0│   ──►   │y0│ = Σ w0j·xj  ← 行0 点乘 x
//   行1 │ w10 w11 .. w1,n-1│   │x1│         │y1│ = Σ w1j·xj
//    .. │      ...         │   │..│         │..│
//  行d-1│ ...              │   │xn-1│       │yd-1│
//       └──────────────┘       └──┘         └──┘
//        每一行 → 输出的一个维度;d 行 → d 维输出。共 d×n 次乘加。
//
//   维度变化:n 维输入 → d 维输出(d 可以 = / > / < n,取决于这一层想要的输出宽度)。
//
//   为什么调 cuBLAS:矩阵乘是计算量最大、最难写快的部分(本项目里唯一不自写 kernel 的),
//   直接用 NVIDIA 高度优化的库。Sgemv = Single-precision GEneral Matrix-Vector multiply。
//
//   关于转置 CUBLAS_OP_T:权重在内存里按 (n,d) 行主序存(W[j*d + i]),而我们要的是
//   xout = W^row · x,cuBLAS 列主序视角下需要转置才对得上,故传 CUBLAS_OP_T。
// ============================================================================
#ifdef USE_CUDA
// Use cuBLAS for matmul to leverage this included, high-performance library.
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // W is stored in this order: (n=0,d=0), (n=1,d=0), (n=2,d=0), ...
    // so W is n x d in cublas terms & we'll need to transpose.
    // Sgemv does y = alpha * op(A) * x + beta * y (modifying y)
    //   where op can transpose the matrix A
    // Translating to our local vars, that is
    // xout = 1.0*op(w)*x + 0.0*xout
    float alpha = 1.0f;
    float beta = 0.0f; // when this is 0, xout will not be used for input
    cublasSgemv(g_cublas_handle, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, xout, 1);
}
#else
void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}
#endif

// ============================================================================
// RoPE(Rotary Position Embedding,旋转位置编码)
//
// 在做什么:把 token 的"位置"信息【旋转】进 q、k 向量。注意力靠 q·k 点积比较,但点积
//   不看顺序;RoPE 把向量每两个数看成一根"针",按位置把针转一个角度——位置就编码进去了。
//   妙处:转完后两个词的 q·k 只取决于它们的【位置差】→ 注意力天然获得"相对距离"感。
//   ★ 长度不变、只转方向;无矩阵乘、无可学习参数(freq 由公式固定算出)。
//
// 输入: pos(当前token位置), sq=q[288], sk=k[288], kv_dim=288, head_size=48
// 输出: 原地旋转 q、k(长度不变,方向带上位置 pos)
// 启动: <<<1, dim/2=144>>>  —— 144 个线程,每线程转一对 (v[i], v[i+1]),互不干扰
//
//   q[288] 两两配对 = 144 对,按头切分(每头48维=24对):
//    线程: t0 t1 ...t23 │ t24..t47 │ ... │ t120..t143
//          └─ 头0(24对)─┘└─ 头1 ──┘     └── 头5 ──┘
//
//   每个线程做 5 步:
//    ① i=t*2; head_dim=i%head_size                  定位这对、算频率编号
//    ② freq = 1/10000^(head_dim/head_size)           转速:head_dim小→freq≈1(快,管近)
//                                                          head_dim大→freq≈0.0002(慢,管远)
//    ③ θ = pos * freq                                角度:pos越大转越多(位置注入处)
//    ④ cosθ, sinθ
//    ⑤ 2D 旋转,原地写回(对 q;若 i<kv_dim 也对 k):
//         v1 ^   ╱原针(v0,v1)                 vec[i]   = v0·cosθ − v1·sinθ  (新v0)
//            │  ╱  ╲ 转θ后(长度不变)          vec[i+1] = v0·sinθ + v1·cosθ  (新v1)
//            └────→ v0
//
//   频率谱(一个头内24对,高频→低频,像钟表 秒针→时针 同时编码近/远距离):
//     freq 1.0┤█▇▆▅▄▃▂▁▁▁ ▁  ▁   ▁    ▁
//          0.0┼──────────────────────────► 对编号(0→23)
//             近距离敏感 ←────→ 远距离敏感
// ============================================================================
#ifdef USE_CUDA
__global__ void RoPe_rotation_kernel(int pos, float *sq, float *sk, int kv_dim, int head_size) {
    int i = threadIdx.x * 2;        // ① 这对的下标 (i, i+1);线程 t 管第 t 对
    int head_dim = i % head_size;   //    在头内排第几(决定频率;每头循环一遍)
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);  // ② 转速
    float val = pos * freq;         // ③ 旋转角度 θ = 位置 × 频率
    float fcr = cosf(val);          // ④ cosθ
    float fci = sinf(val);          //    sinθ
    int rotn = i < kv_dim ? 2 : 1;  // 转几个:2=q和k都转;1=只转q(GQA时靠后的对k已无对应)
    for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? sq : sk; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;   // ⑤ 2D 旋转:新 v0
        vec[i+1] = v0 * fci + v1 * fcr;   //            新 v1(长度不变)
    }
}
void RoPe_rotation(int pos, RunState* s, int dim, int kv_dim, int head_size) {
    RoPe_rotation_kernel <<<1, dim/2 >>> (pos, s->q, s->k, kv_dim, head_size);
}
#else
void RoPe_rotation(int pos, RunState* s, int dim, int kv_dim, int head_size) { //s->q, s->k, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size) {
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}
#endif

// ============================================================================
// 多头自注意力(Multi-Head Self-Attention)—— forward 第 4 步,Transformer 的核心
//
// 【目的】让"当前 token"回看自己和所有历史 token,按相关度从它们身上提取信息,
//         得到一个"融合了上下文"的新表示。这是模型理解上下文、抓长距离依赖的关键。
//
// 【输入】
//   pos          : 当前 token 的位置(只能看 0..pos,看不到未来 = 因果性)
//   sq = q[288]  : 当前 token 的 query(已 RoPE,带位置信息)
//   key_cache    : 历史所有 token 的 K(KV Cache 里读,不重算)
//   value_cache  : 历史所有 token 的 V
//   head_size : 每个头的维度 = dim/n_heads = 48。把 288 切成 6 头各 48 维,点积在48维内算。
//   kv_mul    : 几个 Q 头共享一组 K/V = n_heads/n_kv_heads。本模型 6/6=1(MHA,各读各的);
//               GQA 时 >1,如 kv_mul=2 → Q0,Q1 共用 KV0。映射:第 h 个Q头读第 (h/kv_mul) 个KV头。
//   loff      : 本层在 KV Cache 的起始偏移 = l*seq_len*kv_dim,用来定位"第几层"。
//
//   这三个 + t*kv_dim 在 KV Cache 大数组里【三级定位】一个 K/V 向量:
//     k = key_cache + loff + t*kv_dim + (h/kv_mul)*head_size
//                      │        │              │
//                   定位层l   定位第t个token  定位第h头(切48维)
//     ┌──── 层0 ────┬─ 层1 ─┬...┐   每层内:[pos0][pos1]...   每个pos内:[头0|头1|...|头5]
//     └─ +loff 跳到第l层 ──────┘    └ +t*kv_dim 跳到第t个 ┘   └ +(h/kv_mul)*48 跳到第h头 ┘
//
// 【输出 / 结果存哪】
//   写入 sxb(即 RunState.xb)[288]:当前 token 看完上下文后的新表示。
//   随后被第 5 步 Wo 投影,再残差加回主干 x。
//
// 【做什么计算】具体例子:序列 "The cat sat on",当前算 "on"(pos=3),回看 0..3。
//   每个头独立做 3 步(q 和历史 k/v 都按头切成 48 维):
//
//   ① 打分:当前 q 和每个历史 k 点积,÷√head_size
//        q·k0=1.2  q·k1=3.5  q·k2=0.8  q·k3=2.0
//        att = [ 1.2 , 3.5 , 0.8 , 2.0 ]      原始分(可正负、无范围)
//                            │
//   ② softmax → 权重(放大高分、压低低分,和=1):
//        w   = [ 0.08, 0.65, 0.05, 0.22 ]     ← "on" 最该关注 "cat"(0.65)
//                The   cat   sat   on
//                            │
//   ③ 加权求和:按权重把历史的 v 混合(像按比例调鸡尾酒):
//        w[0]·v0(The) ┐
//        w[1]·v1(cat) ├─ 全部相加 ─► xb(本头48维)= 看完上下文的新表示
//        w[2]·v2(sat) │              (cat 的 v 占比最大 → 输出主要带上 cat 的信息)
//        w[3]·v3(on)  ┘
//
//   6 个头各看一个"角度"(语法/修饰/远距离…),各做一遍 → 拼成 xb[288]:
//      头0      头1      ...  头5
//      [48维] + [48维] + ... +[48维]  = xb[288]
//
// 【并行】<<<n_heads=6, 1024>>>:每个 block 算一个头(blockIdx.x=h),块内 1024 线程协作:
//      block h:  ① 1024线程分摊 0..pos 个历史位置打分 → __syncthreads
//                ② 块内 1024 线程协作做 softmax(归约求max/求和)→ __syncthreads
//                ③ 1024线程分摊输出的 48 维,各算 xb[i]=Σ att[t]·v[t][i]
//
// 【要点】
//   • 只看 0..pos(循环上界 t<=pos)= 因果/自回归,看不到未来,无需显式 mask。
//   • RoPE 在第①步 q·k 生效 → 点积自动含"两 token 相隔多远"的位置信息。
//   • 历史 k/v 全从 KV Cache 读(不重算)→ 这就是 KV Cache 省算力的地方。
//   • 6 个头关注"角度"不同(语法/语义/距离…),最后拼起来。
// ============================================================================
#ifdef USE_CUDA
// TODO refactor vs C code
// ---- kernel 参数逐个含义 -----------------------------------------------------
//   pos         : 当前 token 位置;本次只对历史 0..pos 做注意力(因果)
//   seq_len     : 序列最大长度(256),是 att / KV Cache 的行跨距
//   sq          : 当前 token 的 query,[n_heads*head_size]=288,已 RoPE
//   satt        : 注意力分数缓冲,[n_heads, seq_len];satt + h*seq_len = 第h头那一行
//   sxb         : 输出缓冲(=RunState.xb),[288];每头把结果写进自己那 48 维
//   key_cache   : 所有历史 token 的 K,布局 [layer][seq_len][kv_dim]
//   value_cache : 所有历史 token 的 V,同布局
//   kv_dim      : 一个位置的 K/V 总维度(288;= head_size*n_kv_heads),也是位置间步长
//   kv_mul      : 几个 Q 头共享一组 KV(=n_heads/n_kv_heads);本模型=1(MHA)
//   head_size   : 每个头的维度(48)
//   loff        : 本层在 KV Cache 的起始偏移(= l*seq_len*kv_dim),定位"第几层"
//
// ---- 每个 block 处理什么任务 -------------------------------------------------
//   grid = n_heads(6),即【一个 block 负责一个注意力头 h = blockIdx.x】。
//   该 block 内 1024 个线程协作,完成"第 h 头"的全部注意力:
//     ① 用本头的 q(48维)和历史每个 token 的本头 k 打分 → 填 att[0..pos]
//     ② 对 att[0..pos] 做 softmax 得权重
//     ③ 按权重对历史本头的 v 加权求和 → 写进 sxb 第 h 头那 48 维
//   6 个 block 并行 = 6 个头同时算,互不干扰;拼起来就是完整的 xb[288]。
// -----------------------------------------------------------------------------
__global__ void multi_head_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache, float *value_cache, int kv_dim, int kv_mul, int head_size, int loff) {
    int h = blockIdx.x;             // 当前 Q 头编号 (0 .. n_heads-1);本 block 只管这一个头
    // 取本头的 query:q 向量第 h 头那 48 维(从 h*head_size 开始)
    float* q = sq + h * head_size;
    // 本头的分数行:satt 是 [n_heads, seq_len],第 h 头用第 h 行(h*seq_len 起)
    float* att = satt + h * seq_len;
    // —— ① 打分:线程分摊 0..pos 个历史位置,各算 q·k 存入 att[t] ——
    // 【外层循环】迭代"历史 token 的位置 t"(0..pos):每次迭代算"当前token 对 第t个历史"
    //   的一个注意力分数。由 1024 线程分摊(pos<256<1024 时基本=每线程管一个位置 t,
    //   多余线程不进循环)。t<=pos 这个上界 = 只看历史、看不到未来(因果性)。
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // ---- GQA/MHA 的头映射就在这里:第 h 个 Q 头去读第 (h/kv_mul) 个 KV 头 ----
        //   MHA(本模型 kv_mul=1):h/1 = h   → Q头h 读 KV头h(1:1,每Q独享一组KV)
        //   GQA(kv_mul=2 为例)   :h/2      → Q0,Q1→KV0;Q2,Q3→KV1(多Q共享一组KV)
        //
        //     Q头:  [Q0][Q1][Q2][Q3][Q4][Q5]      Q头: [Q0 Q1][Q2 Q3][Q4 Q5]
        //   MHA      │   │   │   │   │   │      GQA      └┬─┘  └┬─┘  └┬─┘
        //     KV头: [K0][K1][K2][K3][K4][K5]      KV头:  [K0]  [K1]  [K2]
        //   (h/kv_mul) 把 Q 头索引折算成它该读的 KV 头索引;乘 head_size 跳到该 KV 头起点
        float* k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        // 【内层循环】q·k 点积:在本头的 48 维里逐元素乘加,得到一个分数标量
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {   // head_size=48 次乘加
            score += q[i] * k[i];
        }
        score /= sqrtf(head_size);   // ÷√48 缩放:防维度大时点积过大、softmax 饱和
        // 存进本头分数行的第 t 个位置
        att[t] = score;
    }
    // above was this threads portion of the iteration.  wait for all threads to finish
    __syncthreads();

    // —— ② softmax:把分数变成权重(和为1),att[0..pos] 原地变成注意力权重 ——
    // 注意:att = satt + h*seq_len,所以这里只对【当前 block 负责的第 h 个 head】那一行做
    //   softmax(每个 head 的权重各自和为1,本就按头独立)。6 个 head 由 6 个 block 并行各做各的;
    //   head 内部由本 block 的 1024 线程协作完成(见 softmax_gpu 的三段归约)。
    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax_gpu(att, pos + 1);
    __syncthreads();

    // —— ③ 加权求和:用权重把历史的 V 混合,xb[i] = Σ_t att[t]·v[t][i],写回 xb(本头) ——
    // 注意循环变量变了:① 打分时外层迭代"历史位置 t";这里外层迭代"输出维度 i(0..47)",
    //   即【一个线程负责输出的一维】,内层再遍历所有历史 token 累加。
    // ★ 线程利用率:本步 i<head_size=48,所以 1024 线程里只有【前 48 个】进循环干活,
    //   其余 976 个空转。(各步有效线程数不同:① 打分 ≤ pos+1(最多256);③ 固定 48。
    //    block 仍开 1024 是因为 cub::BlockReduce 要求 blockDim 固定、且要覆盖最大情况。
    //    线程利用率低是朴素实现的典型特征,工业级 flash-attn 会精心排满线程。)
    // weighted sum of the values, store back into xb
    // NOTE: by swapping the order of the for loops (vs. C) a simpler
    // version of the code accomplishes the same task and fits more
    // naturally with the CUDA way of subdividing the problem.
    float* xb = sxb + h * head_size;   // 本头输出写到 xb 的第 h 头那 48 维(h*head_size 起)
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {   // 外层:输出维度 i;仅 i<48 干活
        float val = 0.0f;
        // 本步用到两类数据:
        //   v   ← 外部输入 value_cache(第t个历史的V,本头48维)
        //   att ← 【本kernel内部结果】:第②步 softmax 算出的权重(原地存在 satt,
        //          经①打分→②归一化得来;两次 __syncthreads 保证这里读到的是成品权重)
        for (int t = 0; t <= pos; t++) {   // 内层:遍历所有历史 token,累加 att[t]·v[t][i]
            // 同理:第 h 个 Q 头读第 (h/kv_mul) 个 KV 头的 V(MHA 时即第 h 个)
            float* v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            float a = att[t];        // 第②步 softmax 的权重(注意:已不是原始分数)
            val += a * v[i];         // 权重 × value 累加
        }
        xb[i] = val;   // 写回:本头输出第 i 维(6个头各填48维 → 拼成 xb[288])
    }
}
void multi_head_attention(int pos, Config* p, RunState* s, int kv_dim, int kv_mul, int head_size, int loff) {
    multi_head_attention_kernel <<<p->n_heads, num_threads_lrg>>> (pos, p->seq_len, s->q, s->att, s->xb, s->key_cache, s->value_cache, kv_dim, kv_mul, head_size, loff);
}
#else
void multi_head_attention(int pos, Config* p, RunState* s, int kv_dim, int kv_mul, int head_size, int loff) {
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < p->n_heads; h++) {
        // get the query vector for this head
        float* q = s->q + h * head_size;
        // attention scores for this head
        float* att = s->att + h * p->seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(att, pos + 1);

        // weighted sum of the values, store back into xb
        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
}
#endif

#ifdef USE_CUDA
__global__ void f_silu_elementwise_mul_w3_kernel(float *shb, float *shb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = shb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= shb2[i];
        shb[i] = val;
    }
}
void f_silu_elementwise_mul_w3(RunState *s, int hidden_dim) {
    f_silu_elementwise_mul_w3_kernel<<<divUp(hidden_dim, num_threads_med), num_threads_med>>>(s->hb, s->hb2, hidden_dim);
}
#else
void f_silu_elementwise_mul_w3(RunState *s, int hidden_dim) {
    for (int i = 0; i < hidden_dim; i++) {
        float val = s->hb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= s->hb2[i];
        s->hb[i] = val;
    }
}
#endif

#ifdef USE_CUDA
__global__ void accum_kernel(float* a, float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}
void accum(float *a, float *b, int size) {
    accum_kernel<<<divUp(size, num_threads_med), num_threads_med>>>(a,b,size);
}
#else
void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}
#endif

float* forward(Transformer* transformer, int token, int pos) {

    // ====================== 便捷别名(只为少打字,无计算)======================
    Config* p = &transformer->config;            // 超参蓝图
    TransformerWeights* w = &transformer->weights;// 权重(显存)
    RunState* s = &transformer->state;            // 激活缓冲(草稿纸)
    float *x = s->x;                              // 主干隐藏状态,贯穿全程,长度=dim

    // ====================== 关键维度量(后面反复用)==========================
    int dim        = p->dim;                      // 288  主干维度 = token 向量长度
    int hidden_dim = p->hidden_dim;               // 768  FFN 内部膨胀维度(升维→激活→降回)
    int head_size  = dim / p->n_heads;            // 48   每个注意力头的维度 = dim / n_heads

    // --- 下面两个是为 GQA(分组查询注意力)通用性准备的 ---
    // kv_dim:一个位置的 K(或 V)向量总长度 = head_size × n_kv_heads
    //         = 所有 KV 头拼起来的总维度。代码写法 (dim*n_kv_heads)/n_heads 与之等价。
    //   本模型(MHA):head_size=48, n_kv_heads=6 → kv_dim = 48×6 = 288 (= dim)
    //   普通 MHA:n_kv_heads == n_heads → kv_dim == dim
    //   GQA     :n_kv_heads <  n_heads → kv_dim <  dim(KV头更少,K/V投影和KV Cache都更省)
    //     例:Llama2-70B head_size=128, n_kv_heads=8 → kv_dim=128×8=1024 < dim=8192
    //   注:Q 永远是 dim(288=6头×48);只有 K/V 用 kv_dim。
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    // kv_mul:多少个 Q 头共用一组 K/V = n_heads / n_kv_heads。
    //   MHA → 1(每个 Q 头独享自己的 K/V);GQA → >1(几个 Q 头挤一组 K/V)
   int kv_mul = p->n_heads / p->n_kv_heads;
    //
    //        n_heads=6 个 Q 头                 n_kv_heads 个 KV 头
    //   MHA  [Q0][Q1][Q2][Q3][Q4][Q5]    →    [KV0][KV1][KV2][KV3][KV4][KV5]  kv_mul=1
    //   GQA  [Q0 Q1][Q2 Q3][Q4 Q5]       →    [ KV0 ][ KV1 ][ KV2 ]           kv_mul=2
    //         └每2个Q头共享1组KV┘               (本模型是上面的 MHA:kv_dim=288, kv_mul=1)

    // 用 token id 做"行索引",在词嵌入表里定位该 token 那一行 embedding 的起始指针。
    // token_embedding_table 逻辑上是 [vocab, dim],内存里一维连续铺开;按行寻址:
    // 第 i 行起点 = 基址 + i*dim(经典二维数组寻址)。这是每步唯一一次"喂数据上 GPU"。
    //
    //   token_embedding_table  (一维连续, 逻辑 32000 × 288)
    //   ┌───────┬───────┬───────┬─────┬────────┐   每行 = dim = 288 个 float
    //   │ 行 0  │ 行 1  │ 行 2  │ ... │行 31999│
    //   └───────┴───────┴───────┴─────┴────────┘
    //   ▲                       ▲
    //   基址              +token*dim ──► content_row(第 token 行开头)
    //                            └─ 这 288 个 float 就是该 token 的 embedding
    //
    // 注意:这里只是"算地址"(指针偏移,不搬数据);真正复制由下面的 memcpy/cudaMemcpy 完成。
    float* content_row = w->token_embedding_table + token * dim;
#ifdef USE_CUDA
    // 注:content_row 实为【显存】指针(token_embedding_table 已被 read_checkpoint
    // 当作权重区一部分拷到显存)。故此处实质是 显存→显存 取一行到 x;cudaMemcpyKind
    // 写成 HostToDevice 不严谨(应为 DeviceToDevice),靠 CUDA UVA 自动识别才能跑通。
    CUCHK(cudaMemcpy(x, content_row, dim*sizeof(*x), cudaMemcpyHostToDevice));
#else
    memcpy(x, content_row, dim*sizeof(*x));
#endif

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // 第 1 步:注意力前 RMSNorm。输入 x[288],用本层增益 rms_att_weight(+l*dim 定位第l层),
        // 输出归一化后的 xb[288]。x 本身不变(残差要用),结果写到 xb 供 QKV 投影。详见 rmsnorm_kernel。
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // ---- 在 KV Cache 这块大数组里定位"当前层 l、当前位置 pos"的 K/V 槽位 ----
        // KV Cache 内存布局是 [layer][seq_len][kv_dim],定位要两级偏移:
        //
        //   key_cache: ┌──────── 层0 ────────┬──── 层1 ────┬ ... ┬─ 层(L-1) ─┐
        //              │pos0 pos1 ... pos(S-1)│             │     │           │
        //              └──────────────────────┴─────────────┴─────┴───────────┘
        //               │← seq_len × kv_dim ─→│  每层这么大
        //
        //   loff          = l * seq_len * kv_dim   → 先跳过前面 l 层,定位到第 l 层起点
        //   + pos * kv_dim                          → 再在本层内跳到第 pos 个 token 的槽位
        int loff = l * p->seq_len * kv_dim;       // 第 l 层在 KV Cache 里的起始偏移
        s->k = s->key_cache + loff + pos * kv_dim;   // 当前(层l,位置pos)的 K 写入点
        s->v = s->value_cache + loff + pos * kv_dim; // 当前(层l,位置pos)的 V 写入点
        // 接着的 matmul(s->k, ...) 算出的 K 会直接落进这个槽位 = 同时完成"算K"和"缓存K"

        // ======================= 第 2 步:QKV 投影 ===============================
        // 用三个权重矩阵,把归一化后的隐藏状态 xb[288] 投影成 Query / Key / Value 三个向量。
        // 这是注意力的"提问/索引/内容"三件套:
        //   Q (Query 查询):我(当前token)想找什么信息    —— 拿去和别人的 K 打分
        //   K (Key   键)  :我能被别人按什么"标签"检索到  —— 写入 KV Cache 供以后查
        //   V (Value 值)  :真正被取走的内容              —— 写入 KV Cache,按注意力权重加权
        //
        //   每个都是一次 matmul(矩阵×向量,见 matmul 注释):
        //        Wq[288×288] · xb[288] → q[288]      (Q 用 dim)
        //        Wk[kv_dim×288]· xb[288] → k[kv_dim]  (K 用 kv_dim;MHA 时=288)
        //        Wv[kv_dim×288]· xb[288] → v[kv_dim]  (V 用 kv_dim)
        //
        //   注意:输入都是同一个 xb,但乘三个【不同】的权重 → 得到三个不同向量。
        //
        //   ── Wq/Wk/Wv 长什么样:每层一个【288×288 的二维矩阵】(不是向量!)──────────
        //     与 RMSNorm 的 weight 对比:
        //       RMSNorm weight: [g0 g1 ... g287]        1维向量, 288个数, 逐元素乘
        //       Wq:             ┌ w00 w01 ... w0,287 ┐
        //                       │ w10 ...            │  2维矩阵, 288×288=82944个数, 矩阵乘
        //                       └ w287,0 ... w287,287┘
        //     真实数值(stories15M 第0层 Wq):一堆有正有负的小数,多在 ±0.1 内,
        //       均值≈0,std≈0.06,例如 [0.0396, -0.0631, 0.0705, ...] —— 训练学出的稠密矩阵。
        //     元素含义:W[i][j] = 输入第 j 维 对 输出第 i 维 的贡献权重。
        //
        //   "多头"是逻辑切分:q[288] 其实是 6 个头 ×48 拼在一起,后续按 head_size 切开:
        //        q: [ 头0(48) | 头1(48) | 头2 | 头3 | 头4 | 头5(48) ]  = 288
        //     即 Wq 的不同"行段"生成不同头的 Query(前48行→头0,接着48行→头1,...)。
        //   K/V 同理(MHA 时也是 6 头;GQA 时 KV 头更少,见 kv_dim/kv_mul 说明)。
        //
        //   +l*dim*dim / +l*dim*kv_dim:在权重大块里定位【第 l 层】的 Wq/Wk/Wv(type-major 布局)。
        //   s->k / s->v 已指向 KV Cache 当前槽位 → 算完即缓存(见上方 KV Cache 定位注释)。
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        RoPe_rotation(pos, s, dim, kv_dim, head_size);

        // 第 4 步:多头自注意力。输入当前 q(已RoPE) + KV Cache 里 0..pos 的历史 k/v,
        // 每头做 打分→softmax→加权求V,结果写回 s->xb[288](= 看完上下文的新表示)。
        // 详见 multi_head_attention_kernel 注释。
        //   注意力的输出 = s->xb[288](6个头各48维拼成),作为下面两步的输入。
        multi_head_attention(pos, p, s, kv_dim, kv_mul, head_size, loff);

        // ===== 第 5 步:Wo 输出投影 =====================================================
        // 【Wo 是什么】注意力的"出口投影"矩阵。在模型文件的权重区里(随权重一起 mmap 到显存),
        //   是【每层独立、训练好固定】的 288×288 稠密矩阵(6层=6份;非向量)。权重布局日志:
        //     wo (layer,Qd,dim) 总元素497664 = 6×82944 → 每层 288×288=82944,各层不同。
        //
        // 【目的】注意力第③步的 xb = 6个头输出的【简单拼接】,各头之间还没"交流"。Wo 做线性变换:
        //   xb2[i] = Σ_j wo[i][j]·xb[j] → 输出每一维都混合了全部288输入(=所有头)→【融合多头】;
        //   同时输出仍是288维 → 【变回主干空间】,以便残差加回 x。
        //   类比:6个头=6份独立报告,Wo=汇总者把它们交叉综合成一份统一结论。
        //
        // 【matmul 各参数】matmul(xout, x, w, n, d):
        //   s->xb2              = xout : 输出,注意力子块的最终结果 [288]
        //   s->xb               = x    : 输入,多头注意力的拼接输出 [288](第④步产物)
        //   w->wo + l*dim*dim   = w    : 权重,第 l 层的 Wo 矩阵([288×288];+l*82944 定位本层)
        //   dim (=288)          = n    : 输入维度
        //   dim (=288)          = d    : 输出维度(288→288,不变维)
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);   // xb → xb2(融合多头)

        // ===== 第 6 步:残差连接 ========================================================
        // 【在干什么】算完注意力后,不把 x 替换成新结果,而是把注意力结果 xb2【加回】原来的 x:
        //     x = x + xb2
        //   注意力始终没改 x(中间结果都写在 xb/xb2),这里才用【原始 x】相加。
        //
        //     x ─┬───────────────► (一条捷径,x 原样保留)
        //        │                          │
        //        └─►注意力计算→xb2          ▼
        //                        x = x + xb2 (把结果加回 x)
        //
        // 【三个向量是什么】(都是 288 维)
        //   x   原先 = 这个 token 进入本层时的表示(注意力全程没动它,还是本层入口的干净值)
        //   xb2       = 注意力+Wo 算出的"上下文增量Δ"(回看前文后该补充进来的信息)
        //   x   加完 = 原表示 + 上下文信息 = 更懂上下文的新表示
        //     例:token "on" 原本只是"on",加完后 x 里融入了前文 cat/sat → "在cat sat语境下的on"
        //       x原先:[x0,  x1,  x2, ...]
        //       xb2:  [d0,  d1,  d2, ...]   逐元素相加
        //       x新:  [x0+d0, x1+d1, ...]
        //
        // 【加完的 x 后续怎么用】这个新 x 立刻被本层后面的 FFN 使用,再传给下一层:
        //   词嵌入=x
        //    第0层: x ─+注意力Δ─► x ─+FFN Δ─► x ┐
        //    第1层: x ─+注意力Δ─► x ─+FFN Δ─► x │  每层在 x 上"加两次",越加越精炼
        //     ...                               │
        //    第5层: x ─+注意力Δ─► x ─+FFN Δ─► x ┘
        //    最终:  norm → lm_head → 预测下一个词(用最终的 x)
        //   ↑ 本行加完的 x → 紧接着 ffn rmsnorm(下面)就用它;FFN算完再加一次;
        //     本层结束的 x → 成为下一层开头的 x。x 就是这条贯穿所有层的主线(残差流)。
        //
        // 【输入】原始 x[288] + 注意力结果 xb2[288]  【输出】新的 x = x+xb2(原地写回)
        // 【为什么加回去而不是替换】
        //   ① 给信息/梯度一条直达高速路 → 很深的网络也能训练(最重要;否则梯度逐层消失)。
        //   ② 每层只需学"在 x 上改动一点"(增量),不必"重造"整个表示,任务更简单。
        //   ③ 原始信息不丢:就算这次计算没用(结果≈0),x 也原样传下去。
        accum(x, s->xb2, dim);                                 // x = x + xb2

        // ===== 第 7 步:FFN 前的 RMSNorm(Pre-Norm)=====================================
        // 【在做什么】进入前馈网络(FFN)之前,先把 x 归一化。和第①步的注意力前 RMSNorm
        //   完全一样(同一 kernel),只是位置在 FFN 之前、用本层的 rms_ffn_weight。
        //   体现 Pre-Norm:每次大计算(注意力/FFN)前都先归一化以稳定数值。
        //
        // 【输入】x[288]   :注意力残差后的隐藏状态(刚被 x=x+xb2 更新过)
        //         rms_ffn_weight + l*dim : 本层 FFN 的归一化增益[288](每层独立)
        // 【输出】xb[288]  :归一化结果,作为下面 FFN 的输入(x 本身不动)
        //
        //   x(注意力残差后)
        //     │ rmsnorm(× rms_ffn_weight)        ← x 不被覆盖(FFN后的残差还要用原始x)
        //     ▼
        //   xb ──► 喂给 FFN(W1/W3 升维 → SwiGLU → W2 降维)
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);   // x → xb(FFN 输入)

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        f_silu_elementwise_mul_w3(s, hidden_dim);

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // 残差连接(同第6步):把 FFN 结果加回原始 x。x = x + FFN输出。
        //   一层里共两次残差:注意力后一次、FFN 后一次。x 一路被加着精炼 → 进下一层。
        accum(x, s->xb, dim);   // x = x + FFN结果
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
#ifdef USE_CUDA
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);
    CUCHK(cudaMemcpy(s->logits, s->logits_gpu, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
#else
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
#endif 
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
#if defined USE_CUDA && defined _WIN32
    // CUDA on Windows was not capable of handling the syntax below
    TokenIndex tok;
    tok.str = str;
#else
    TokenIndex tok = { .str = str }; // acts as the key to search for
#endif
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// [观测] 把当前 token 序列以可读片段形式打印出来,空格显示为 '_' 方便看
static void trace_bpe_seq(Tokenizer* t, int* tokens, int n) {
    fprintf(stderr, "        ");
    for (int i = 0; i < n; i++) {
        const char* s = t->vocab[tokens[i]];
        fprintf(stderr, "[");
        for (const char* p = s; *p; p++) fputc(*p == ' ' ? '_' : *p, stderr);
        fprintf(stderr, "]");
    }
    fprintf(stderr, "   (%d tokens)\n", n);
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
    int trace = (getenv("TRACE_BPE") != NULL); // [观测] 设了环境变量才打印 BPE 过程

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char *)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // [观测] 打印 BPE 合并前的初始拆分
    if (trace) {
        fprintf(stderr, "\n========== [encode] BPE 合并过程: \"%s\" ==========\n", text);
        fprintf(stderr, "  初始拆分(每个字符/字节一个 token):\n");
        trace_bpe_seq(t, tokens, *n_tokens);
        fprintf(stderr, "  开始合并(每轮选 score 最高的相邻对):\n");
    }
    int merge_step = 0;

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // [观测] 打印这一步合并了哪一对、合成什么、score 多少
        if (trace) {
            char a[256], b[256];
            snprintf(a, sizeof(a), "%s", t->vocab[tokens[best_idx]]);
            snprintf(b, sizeof(b), "%s", t->vocab[tokens[best_idx+1]]);
            for (char* p=a; *p; p++) if (*p==' ') *p='_';
            for (char* p=b; *p; p++) if (*p==' ') *p='_';
            char merged[512]; snprintf(merged, sizeof(merged), "%s", t->vocab[best_id]);
            for (char* p=merged; *p; p++) if (*p==' ') *p='_';
            fprintf(stderr, "  第%2d步: 合并 [%s]+[%s] -> [%s]  (id=%d, score=%.4f, 位置%d)\n",
                    ++merge_step, a, b, merged, best_id, best_score, best_idx);
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
        if (trace) trace_bpe_seq(t, tokens, *n_tokens);
    }

    if (trace) {
        fprintf(stderr, "  最终 %d 个 token,id 序列: [", *n_tokens);
        for (int i = 0; i < *n_tokens; i++) fprintf(stderr, "%s%d", i?", ":"", tokens[i]);
        fprintf(stderr, "]\n========================================================\n\n");
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = (char *)"";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = (char *)"tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = (char *)"generate";    // generate|chat
    char *system_prompt = (char *)NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

#ifdef USE_CUDA
    create_cublas_handle();
#endif

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
#ifdef USE_CUDA
    destroy_cublas_handle();
#endif
    return 0;
}
#endif
