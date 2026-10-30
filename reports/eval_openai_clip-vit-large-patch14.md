**Modelo:** `openai/clip-vit-large-patch14` · **Índice:** 1.999 produtos

#### imagem → imagem (500 consultas)

| Métrica | Resultado (IC 95%) | Acaso | Ganho |
|---|---|---|---|
| P@1 (tipo) | 0.778 ± 0.036 | 0.055 | 14.1× |
| P@5 (tipo) | 0.736 ± 0.028 | 0.055 | 13.3× |
| P@10 (tipo) | 0.694 ± 0.028 | 0.055 | 12.6× |
| MRR (tipo) | 0.844 ± 0.027 | - | - |
| P@5 (categoria) | 0.982 ± 0.009 | 0.340 | 2.9× |
| P@5 (tipo+cor) | 0.208 ± 0.021 | 0.008 | 27.7× |

#### texto → imagem (96 consultas)

| Métrica | Resultado (IC 95%) | Acaso | Ganho |
|---|---|---|---|
| P@5 (tipo) | 0.702 ± 0.063 | 0.050 | 14.1× |
| P@10 (tipo) | 0.686 ± 0.058 | 0.050 | 13.7× |
| P@5 (tipo+cor) | 0.435 ± 0.063 | 0.007 | 63.9× |
| P@10 (tipo+cor) | 0.406 ± 0.051 | 0.007 | 59.6× |
| MRR (tipo+cor) | 0.659 ± 0.076 | - | - |
| HitRate@10 (tipo+cor) | 0.917 ± 0.056 | - | - |

#### texto → descrição (96 consultas)

| Métrica | Resultado (IC 95%) | Acaso | Ganho |
|---|---|---|---|
| P@5 (tipo) | 0.752 ± 0.060 | 0.050 | 15.1× |
| P@10 (tipo) | 0.667 ± 0.055 | 0.050 | 13.4× |
| P@5 (tipo+cor) | 0.654 ± 0.060 | 0.007 | 96.0× |
| P@10 (tipo+cor) | 0.520 ± 0.052 | 0.007 | 76.3× |
| MRR (tipo+cor) | 0.872 ± 0.058 | - | - |
| HitRate@10 (tipo+cor) | 0.958 ± 0.040 | - | - |

_Relevância definida pelos metadados do catálogo (mesmo tipo de artigo; estrito = tipo e cor). "Acaso" é a precisão esperada de um ranking aleatório._
