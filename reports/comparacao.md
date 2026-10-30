**Modelos:** `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14` · **Índice:** 1999 produtos

#### imagem → imagem

| Métrica | openai/clip-vit-base-patch32 | openai/clip-vit-large-patch14 | Melhor | Separado? |
|---|---|---|---|---|
| P@1 (tipo) | 0.748 ± 0.038 | 0.778 ± 0.036 | openai/clip-vit-large-patch14 | não |
| P@5 (tipo) | 0.706 ± 0.030 | 0.736 ± 0.028 | openai/clip-vit-large-patch14 | não |
| P@10 (tipo) | 0.682 ± 0.029 | 0.694 ± 0.028 | openai/clip-vit-large-patch14 | não |
| MRR (tipo) | 0.819 ± 0.029 | 0.844 ± 0.027 | openai/clip-vit-large-patch14 | não |
| P@5 (categoria) | 0.981 ± 0.009 | 0.982 ± 0.009 | openai/clip-vit-large-patch14 | não |
| P@5 (tipo+cor) | 0.232 ± 0.023 | 0.208 ± 0.021 | openai/clip-vit-base-patch32 | não |

#### texto → imagem

| Métrica | openai/clip-vit-base-patch32 | openai/clip-vit-large-patch14 | Melhor | Separado? |
|---|---|---|---|---|
| P@5 (tipo) | 0.665 ± 0.070 | 0.702 ± 0.063 | openai/clip-vit-large-patch14 | não |
| P@10 (tipo) | 0.635 ± 0.064 | 0.686 ± 0.058 | openai/clip-vit-large-patch14 | não |
| P@5 (tipo+cor) | 0.462 ± 0.066 | 0.435 ± 0.063 | openai/clip-vit-base-patch32 | não |
| P@10 (tipo+cor) | 0.400 ± 0.051 | 0.406 ± 0.051 | openai/clip-vit-large-patch14 | não |
| MRR (tipo+cor) | 0.631 ± 0.079 | 0.659 ± 0.076 | openai/clip-vit-large-patch14 | não |
| HitRate@10 (tipo+cor) | 0.896 ± 0.061 | 0.917 ± 0.056 | openai/clip-vit-large-patch14 | não |

#### texto → descrição

| Métrica | openai/clip-vit-base-patch32 | openai/clip-vit-large-patch14 | Melhor | Separado? |
|---|---|---|---|---|
| P@5 (tipo) | 0.648 ± 0.060 | 0.752 ± 0.060 | openai/clip-vit-large-patch14 | não |
| P@10 (tipo) | 0.575 ± 0.051 | 0.667 ± 0.055 | openai/clip-vit-large-patch14 | não |
| P@5 (tipo+cor) | 0.537 ± 0.059 | 0.654 ± 0.060 | openai/clip-vit-large-patch14 | não |
| P@10 (tipo+cor) | 0.426 ± 0.046 | 0.520 ± 0.052 | openai/clip-vit-large-patch14 | não |
| MRR (tipo+cor) | 0.754 ± 0.069 | 0.872 ± 0.058 | openai/clip-vit-large-patch14 | não |
| HitRate@10 (tipo+cor) | 0.969 ± 0.035 | 0.958 ± 0.040 | openai/clip-vit-base-patch32 | não |

_"Separado" = os intervalos de confiança de 95% do melhor e do segundo não se sobrepõem._
