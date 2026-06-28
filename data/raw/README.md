# Raw data

`american_bankruptcy.csv` is the unmodified file downloaded from the Kaggle dataset
[American Companies Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset).
The Kaggle dataset page identifies its license as CC0: Public Domain.

The data originate from the dataset accompanying:

> Lombardo, G., Pellegrino, M., Cagnoni, S., & Poggi, A. (2022). Machine Learning for
> Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks.
> *Future Internet, 14*(8), 244. https://doi.org/10.3390/fi14080244

## Integrity information

- Observations: 78,682 firm-years plus one header row
- File size: approximately 11 MB
- SHA-256: `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`

The raw file must remain unchanged. Reproducible pipeline tasks will write transformed
data to `data/interim/` and `data/processed/`.

## Important target note

`status_label` records a company's eventual status and is repeated over its available
history. It is not used directly as a row-level prediction target. A later pipeline step
will reconstruct the one-year-ahead bankruptcy event according to the definition in the
original paper.

