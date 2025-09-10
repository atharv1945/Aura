# sparse_utils.py
def get_sparse_row(sparse_matrix, i):
    coo = sparse_matrix.tocoo()
    row_data = [(r, c, v) for r, c, v in zip(coo.row, coo.col, coo.data) if r == i]
    if not row_data:
        return {}
    return {int(col): float(val) for _, col, val in row_data}
