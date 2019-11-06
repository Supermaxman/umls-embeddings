from typing import get_type_hints
from dataclasses import dataclass, field


def UmlsColumn(col_order, tp):
    def new_type(x):
        return x
    new_type.__name__ = 'UmlsColumn'
    new_type.__supertype__ = tp
    new_type.__col_order = col_order
    return new_type


def get_umls_columns(umls_type):
  col_list = [(x, y) for x, y in get_type_hints(umls_type).items() if y.__name__ == 'UmlsColumn']
  cols = sorted(col_list, key=lambda x: x[1].__col_order)
  umls_cols = [x for x, y in cols]
  return umls_cols


# https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.definitions_file_mrdef_rrf/?report=objectonly
@dataclass
class UmlsDefinition(object):
  cui: UmlsColumn(0, str)
  aui: UmlsColumn(1, str)
  atui: UmlsColumn(2, str)
  satui: UmlsColumn(3, str)
  sab: UmlsColumn(4, str)
  definition: UmlsColumn(5, str)
  suppress: UmlsColumn(6, str)
  cvf: UmlsColumn(7, str)
  # relations: list = field(default_factory=list)
  # parents: list = field(default_factory=list)
  # children: list = field(default_factory=list)
  umls_columns = []

  def __str__(self):
    return f'UmlsDefinition(cui={self.cui}, def={self.definition})'

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return isinstance(other, UmlsDefinition) and self.cui == other.cui and self.definition == other.definition

  def __hash__(self):
    return hash((self.cui, self.definition))


UmlsDefinition.umls_columns = get_umls_columns(UmlsDefinition)


# https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly
@dataclass
class UmlsAtom(object):
  cui: UmlsColumn(0, str)
  lat: UmlsColumn(1, str)
  ts: UmlsColumn(2, str)
  lui: UmlsColumn(3, str)
  stt: UmlsColumn(4, str)
  sui: UmlsColumn(5, str)
  ispref: UmlsColumn(6, str)
  aui: UmlsColumn(7, str)
  saui: UmlsColumn(7, str)
  scui: UmlsColumn(7, str)
  sdui: UmlsColumn(7, str)
  sab: UmlsColumn(7, str)
  tty: UmlsColumn(7, str)
  code: UmlsColumn(7, str)
  string: UmlsColumn(7, str)
  srl: UmlsColumn(7, str)
  suppress: UmlsColumn(7, str)
  cvf: UmlsColumn(7, str)
  umls_columns = []

  def __str__(self):
    return f'UmlsAtom(cui={self.cui}, str={self.string})'

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return isinstance(other, UmlsAtom) and self.cui == other.cui and self.string == other.string

  def __hash__(self):
    return hash((self.cui, self.string))


UmlsAtom.umls_columns = get_umls_columns(UmlsAtom)

# https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf/
@dataclass
class UmlsRelation(object):
  cui1: UmlsColumn(0, str)
  aui1: UmlsColumn(1, str)
  stype1: UmlsColumn(2, str)
  rel: UmlsColumn(3, str)
  cui2: UmlsColumn(4, str)
  aui2: UmlsColumn(5, str)
  stype2: UmlsColumn(6, str)
  rela: UmlsColumn(7, str)
  rui: UmlsColumn(8, str)
  srui: UmlsColumn(9, str)
  sab: UmlsColumn(10, str)
  sl: UmlsColumn(11, str)
  rg: UmlsColumn(12, str)
  dir: UmlsColumn(13, str)
  suppress: UmlsColumn(14, str)
  cvf: UmlsColumn(15, str)
  umls_columns = []

  def __str__(self):
    return f'UmlsRelation(cui2={self.cui2}, rel={self.rel} : {self.rela}, cui1={self.cui1})'

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    # TODO determine if this is right for equality
    return isinstance(other, UmlsRelation) \
           and self.cui1 == other.cui1 and self.cui2 == other.cui2 and self.rel == other.rel

  def __hash__(self):
    return hash((self.cui2, self.rel, self.cui1))


UmlsRelation.umls_columns = get_umls_columns(UmlsRelation)
