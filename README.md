# Recomendador Multimodal de Produtos

Sistema de recomendação de produtos por **imagem**, **texto** ou **os dois juntos**. Os embeddings vêm do CLIP (Hugging Face), a busca por similaridade de cosseno roda no ChromaDB e a interface é um app Streamlit.

```
catálogo (CSV + imagens)
   │  scripts/prepare_data.py        limpeza, imagens padronizadas por ID
   ▼
data/processed/catalog.parquet + images/{id}.jpg
   │  scripts/generate_embeddings.py CLIP em mini-batches, memmap, retomável
   ▼
data/processed/embeddings/*.npy
   │  scripts/build_index.py         ChromaDB (HNSW, cosseno) + metadados
   ▼
data/vectorstore/                    products_image / products_text
   │  src/recommender/search.py      texto, imagem e busca combinada
   ▼
app.py (Streamlit)  ·  scripts/search_demo.py (CLI)
```

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # preencha KAGGLE_USERNAME e KAGGLE_KEY

python scripts/prepare_data.py                         # Passo 1: baixa e limpa o dataset
python scripts/generate_embeddings.py                  # Passo 2: embeddings (CLIP)
python scripts/build_index.py                          # Passo 3: índice no ChromaDB
streamlit run app.py                                   # Passo 5: interface
```

Para testar rápido, use `--limit 2000` no Passo 1 e rode o Passo 2 sem `--limit`. Todos os scripts são idempotentes e têm `--help`.

Busca pela linha de comando:

```bash
python scripts/search_demo.py --text "black running shoes"
python scripts/search_demo.py --image data/processed/images/1530.jpg --exclude-id 1530 --grid resultado.jpg
python scripts/search_demo.py --image foto.jpg --text "in black" --gender Men --max-price 200
```

## Decisões de projeto

| Decisão | Por quê |
|---|---|
| Vetores normalizados em L2 + índice HNSW em `cosine` | O cosseno vira produto interno e o banco não recalcula normas na consulta. |
| Duas coleções (`products_image`, `products_text`) | Texto→imagem é o uso zero-shot do CLIP; texto→descrição acerta mais em marca e nome exato. |
| Memmap `.npy` + checkpoints | RAM constante independentemente do tamanho do catálogo; uma execução interrompida retoma do último checkpoint. |
| Fingerprint (catálogo, modelo, dimensão) no índice | Impede misturar vetores de modelos diferentes sem aviso. |
| Modelo de consulta lido dos metadados da coleção | A consulta sempre usa o mesmo CLIP que indexou, independentemente do `.env`. |
| Filtros derivados do índice, não do catálogo | As opções mostradas existem de fato na busca, e o deploy dispensa o `catalog.parquet`. |
| Busca combinada = média ponderada normalizada de imagem e texto | Heurística simples do CLIP para refinar ("esta jaqueta, mas em preto"). |

## Limitações conhecidas

- **Preços sintéticos.** O dataset do Kaggle não tem preço; a coluna `price` é gerada de forma reprodutível e marcada com `price_is_synthetic=True`.
- **Texto em inglês.** O CLIP original foi treinado majoritariamente em inglês; consultas em português rendem menos.
- **Scores de texto→imagem são baixos por natureza** (em torno de 0.2 a 0.35 mesmo para bons resultados) e não são comparáveis com os de imagem→imagem.
- O dataset *small* tem imagens de baixa resolução; o dataset completo (`MMREC_KAGGLE_DATASET=paramaggarwal/fashion-product-images-dataset`) dá resultados visualmente melhores, ao custo de ~15 GB.

## Testes

```bash
python -m pytest -q
```

Os testes usam um backend de embeddings falso e o ChromaDB real, então rodam sem GPU e sem baixar o CLIP. Eles cobrem batching, persistência e retomada, indexação, busca, filtros e o app (via `streamlit.testing`).

## Deploy

O app só consulta o índice, então o deploy precisa de: o código, `data/vectorstore/`, `data/processed/images/` e dependências. Pontos de atenção:

- Remova `data/vectorstore/` e `data/processed/images/` do `.gitignore` na branch de deploy e use um subconjunto do catálogo (`--limit`) para manter o repositório pequeno.
- O primeiro start baixa o CLIP (~600 MB) e o carrega em RAM junto com o PyTorch. Planos com ~1 GB de RAM podem não bastar; prefira um host com mais memória (confira os limites atuais do Hugging Face Spaces e do Streamlit Community Cloud).
- Para uma imagem menor em CPU, instale o PyTorch pelo índice CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Exemplo de `Dockerfile` (não testado neste repositório):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```
