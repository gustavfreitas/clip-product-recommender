**Modelo:** `openai/clip-vit-base-patch32` · **Índice:** 1.999 produtos

#### imagem → imagem (500 consultas)

| Métrica | Resultado (IC 95%) | Acaso | Ganho |
|---|---|---|---|
| P@1 (tipo) | 0.748 ± 0.038 | 0.055 | 13.6× |
| P@5 (tipo) | 0.706 ± 0.030 | 0.055 | 12.8× |
| P@10 (tipo) | 0.682 ± 0.029 | 0.055 | 12.4× |
| MRR (tipo) | 0.819 ± 0.029 | - | - |
| P@5 (categoria) | 0.981 ± 0.009 | 0.340 | 2.9× |
| P@5 (tipo+cor) | 0.232 ± 0.023 | 0.008 | 30.9× |

#### texto → imagem (96 consultas)

| Métrica | Resultado (IC 95%) | Acaso | Ganho |
|---|---|---|---|
| P@5 (tipo) | 0.665 ± 0.070 | 0.050 | 13.3× |
| P@10 (tipo) | 0.635 ± 0.064 | 0.050 | 12.7× |
| P@5 (tipo+cor) | 0.462 ± 0.066 | 0.007 | 67.9× |
| P@10 (tipo+cor) | 0.400 ± 0.051 | 0.007 | 58.7× |
| MRR (tipo+cor) | 0.631 ± 0.079 | - | - |
| HitRate@10 (tipo+cor) | 0.896 ± 0.061 | - | - |

#### texto → descrição (96 consultas)

| Métrica | Resultado (IC 95%) | Acaso | Ganho |
|---|---|---|---|
| P@5 (tipo) | 0.648 ± 0.060 | 0.050 | 13.0× |
| P@10 (tipo) | 0.575 ± 0.051 | 0.050 | 11.5× |
| P@5 (tipo+cor) | 0.537 ± 0.059 | 0.007 | 78.9× |
| P@10 (tipo+cor) | 0.426 ± 0.046 | 0.007 | 62.5× |
| MRR (tipo+cor) | 0.754 ± 0.069 | - | - |
| HitRate@10 (tipo+cor) | 0.969 ± 0.035 | - | - |

_Relevância definida pelos metadados do catálogo (mesmo tipo de artigo; estrito = tipo e cor). "Acaso" é a precisão esperada de um ranking aleatório._
