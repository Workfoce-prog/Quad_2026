
# MN Income Shares + GAI* Demo (Streamlit)

This demo app places a **Minnesota statute-table Income Shares calculator** next to **GAI* / GAI(-CS)** with RAG scoring and a simple simulation lever.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## IMPORTANT
The included MN basic support table CSV is a **template** for demo only.
Replace it with the exact **Minn. Stat. §518A.35 Basic Support Table** values exported into:

- pics_min
- pics_max
- children_count
- bs_total

Upload your exact table in the sidebar to make the Income Shares results exact.
