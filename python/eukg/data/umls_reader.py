import pathlib


def read_psv(line, umls_type):
  line = line.strip()
  data = {}
  if not line:
    return None
  # last pipe element is always empty
  vals = line.split('|')[:-1]
  cols = umls_type.umls_columns
  if len(vals) != len(cols):
    raise ValueError('Line value count does not equal column count!')
  for col_name, col_val in zip(cols, vals):
    data[col_name] = col_val
  return data


def read_umls(umls_path, umls_type, umls_filter=None):
  if not isinstance(umls_path, pathlib.Path):
    umls_path = pathlib.Path(umls_path)
  with umls_path.open('r') as f:
    for line in f:
      data = read_psv(line, umls_type)
      if data is not None:
        item = umls_type(**data)
        if umls_filter is None or umls_filter(item):
          yield item
