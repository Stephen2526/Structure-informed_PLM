"""
Tokenizer class
Modified from TAPE
"""

from lib2to3.pgen2 import token
from typing import List
import logging
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)

IUPAC_CODES = OrderedDict([
    ('Ala', 'A'),
    ('Asx', 'B'),
    ('Cys', 'C'),
    ('Asp', 'D'),
    ('Glu', 'E'),
    ('Phe', 'F'),
    ('Gly', 'G'),
    ('His', 'H'),
    ('Ile', 'I'),
    ('Lys', 'K'),
    ('Leu', 'L'),
    ('Met', 'M'),
    ('Asn', 'N'),
    ('Pro', 'P'),
    ('Gln', 'Q'),
    ('Arg', 'R'),
    ('Ser', 'S'),
    ('Thr', 'T'),
    ('Sec', 'U'),
    ('Val', 'V'),
    ('Trp', 'W'),
    ('Xaa', 'X'),
    ('Tyr', 'Y'),
    ('Glx', 'Z')])

IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])

UNIREP_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("M", 1),
    ("R", 2),
    ("H", 3),
    ("K", 4),
    ("D", 5),
    ("E", 6),
    ("S", 7),
    ("T", 8),
    ("N", 9),
    ("Q", 10),
    ("C", 11),
    ("U", 12),
    ("G", 13),
    ("P", 14),
    ("A", 15),
    ("V", 16),
    ("I", 17),
    ("F", 18),
    ("Y", 19),
    ("W", 20),
    ("L", 21),
    ("O", 22),
    ("X", 23),
    ("Z", 23),
    ("B", 23),
    ("J", 23),
    ("<cls>", 24),
    ("<sep>", 25)])

# Pfam vocab with no ambiguous letters
#'B','X','Z'
PFAM_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("A", 4),
    ("C", 5),
    ("D", 6),
    ("E", 7),
    ("F", 8),
    ("G", 9),
    ("H", 10),
    ("I", 11),
    ("K", 12),
    ("L", 13),
    ("M", 14),
    ("N", 15),
    ("O", 16),
    ("P", 17),
    ("Q", 18),
    ("R", 19),
    ("S", 20),
    ("T", 21),
    ("U", 22),
    ("V", 23),
    ("W", 24),
    ("Y", 25),
    ("B", 26),
    ("Z", 27),
    ("X", 28)])

PFAM_VOCAB_20AA_IDX = [4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,23,24,25]
PFAM_VOCAB_20AA_IDX_MAP = {4:0,5:1,6:2,7:3,8:4,9:5,10:6,11:7,12:8,13:9,14:10,15:11,17:12,18:13,19:14,20:15,21:16,23:17,24:18,25:19,-1:-1,28:-1}

aaCodes = OrderedDict([
    ('Ala', 'A'),
    ('Asx', 'B'),
    ('Cys', 'C'),
    ('Asp', 'D'),
    ('Glu', 'E'),
    ('Phe', 'F'),
    ('Gly', 'G'),
    ('His', 'H'),
    ('Ile', 'I'),
    ('Lys', 'K'),
    ('Leu', 'L'),
    ('Met', 'M'),
    ('Asn', 'N'),
    ('Pro', 'P'),
    ('Gln', 'Q'),
    ('Arg', 'R'),
    ('Ser', 'S'),
    ('Thr', 'T'),
    ('Sec', 'U'),
    ('Val', 'V'),
    ('Trp', 'W'),
    ('Xaa', 'X'),
    ('Tyr', 'Y'),
    ('Glx', 'Z')])

# antibody heavy chain subclass
ab_H_subclass = OrderedDict([("IGHV1",0),("IGHV2",1),("IGHV3",2),("IGHV4",3),("IGHV5",4),("IGHV6",5),("IGHV7",6),("IGHV8",7),("IGHV9",8),("IGHV10",9),("IGHV12",10),("IGHV13",11),("IGHV14",12),("unknown",13)])

ab_L_subclass = OrderedDict([("IGKV1",0),("IGKV1D",1),("IGKV2",2),("IGKV2D",3),("IGKV3",4),("IGKV3D",5),("IGKV4",6),("IGKV5",7),("IGKV6",8),("IGKV6D",9),("IGKV7",10),("IGKV8",11),("IGKV9",12),("IGKV10",13),("IGKV11",14),("IGKV12",15),("IGKV13",16),("IGKV14",17),("IGKV15",18),("IGKV16",19),("IGKV17",20),("IGKV19",21),("IGKV22",22),("IGLV1",23),("IGLV2",24),("IGLV3",25),("IGLV4",26),("IGLV5",27),("IGLV6",28),("IGLV7",29),("IGLV8",30),("IGLV9",31),("IGLV10",32),("unknown",33)])

ab_HL_subclass = OrderedDict([("IGHV1-IGKV1",0),("IGHV1-IGKV10",1),("IGHV1-IGKV11",2),("IGHV1-IGKV12",3),("IGHV1-IGKV13",4),("IGHV1-IGKV14",5),("IGHV1-IGKV15",6),("IGHV1-IGKV16",7),("IGHV1-IGKV17",8),("IGHV1-IGKV19",9),("IGHV1-IGKV1D",10),("IGHV1-IGKV2",11),("IGHV1-IGKV22",12),("IGHV1-IGKV2D",13),("IGHV1-IGKV3",14),("IGHV1-IGKV3D",15),("IGHV1-IGKV4",16),("IGHV1-IGKV5",17),("IGHV1-IGKV6",18),("IGHV1-IGKV8",19),("IGHV1-IGKV9",20),("IGHV1-IGLV1",21),("IGHV1-IGLV2",22),("IGHV1-IGLV3",23),("IGHV1-IGLV4",24),("IGHV1-IGLV5",25),("IGHV1-IGLV6",26),("IGHV1-IGLV7",27),("IGHV1-IGLV8",28),("IGHV2-IGKV1",29),("IGHV2-IGKV10",30),("IGHV2-IGKV12",31),("IGHV2-IGKV13",32),("IGHV2-IGKV19",33),("IGHV2-IGKV1D",34),("IGHV2-IGKV2",35),("IGHV2-IGKV22",36),("IGHV2-IGKV3",37),("IGHV2-IGKV3D",38),("IGHV2-IGKV4",39),("IGHV2-IGKV5",40),("IGHV2-IGKV6",41),("IGHV2-IGKV8",42),("IGHV2-IGKV9",43),("IGHV2-IGLV1",44),("IGHV2-IGLV2",45),("IGHV2-IGLV3",46),("IGHV2-IGLV7",47),("IGHV2-IGLV8",48),("IGHV3-IGKV1",49),("IGHV3-IGKV10",50),("IGHV3-IGKV13",51),("IGHV3-IGKV14",52),("IGHV3-IGKV1D",53),("IGHV3-IGKV2",54),("IGHV3-IGKV22",55),("IGHV3-IGKV2D",56),("IGHV3-IGKV3",57),("IGHV3-IGKV3/OR2",58),("IGHV3-IGKV3D",59),("IGHV3-IGKV4",60),("IGHV3-IGKV5",61),("IGHV3-IGKV6",62),("IGHV3-IGKV6D",63),("IGHV3-IGKV7",64),("IGHV3-IGKV8",65),("IGHV3-IGLV1",66),("IGHV3-IGLV10",67),("IGHV3-IGLV2",68),("IGHV3-IGLV3",69),("IGHV3-IGLV4",70),("IGHV3-IGLV5",71),("IGHV3-IGLV6",72),("IGHV3-IGLV7",73),("IGHV3-IGLV8",74),("IGHV3-IGLV9",75),("IGHV3-unknown",76),("IGHV4-IGKV1",77),("IGHV4-IGKV10",78),("IGHV4-IGKV1D",79),("IGHV4-IGKV2",80),("IGHV4-IGKV2D",81),("IGHV4-IGKV3",82),("IGHV4-IGKV3D",83),("IGHV4-IGKV4",84),("IGHV4-IGKV6",85),("IGHV4-IGKV6D",86),("IGHV4-IGLV1",87),("IGHV4-IGLV2",88),("IGHV4-IGLV3",89),("IGHV4-IGLV4",90),("IGHV4-IGLV5",91),("IGHV4-IGLV6",92),("IGHV4-IGLV7",93),("IGHV4-IGLV8",94),("IGHV4-IGLV9",95),("IGHV4-unknown",96),("IGHV5-IGKV1",97),("IGHV5-IGKV10",98),("IGHV5-IGKV12",99),("IGHV5-IGKV14",100),("IGHV5-IGKV15",101),("IGHV5-IGKV16",102),("IGHV5-IGKV17",103),("IGHV5-IGKV19",104),("IGHV5-IGKV1D",105),("IGHV5-IGKV2",106),("IGHV5-IGKV22",107),("IGHV5-IGKV2D",108),("IGHV5-IGKV3",109),("IGHV5-IGKV4",110),("IGHV5-IGKV5",111),("IGHV5-IGKV6",112),("IGHV5-IGKV8",113),("IGHV5-IGKV9",114),("IGHV5-IGLV1",115),("IGHV5-IGLV2",116),("IGHV5-IGLV3",117),("IGHV5-IGLV6",118),("IGHV5-unknown",119),("IGHV6-IGKV1",120),("IGHV6-IGKV12",121),("IGHV6-IGKV14",122),("IGHV6-IGKV1D",123),("IGHV6-IGKV2",124),("IGHV6-IGKV3",125),("IGHV6-IGKV4",126),("IGHV6-IGKV5",127),("IGHV6-IGKV6",128),("IGHV6-IGKV8",129),("IGHV6-IGLV1",130),("IGHV6-IGLV3",131),("IGHV6-IGLV6",132),("IGHV7-IGKV1",133),("IGHV7-IGKV10",134),("IGHV7-IGKV11",135),("IGHV7-IGKV12",136),("IGHV7-IGKV15",137),("IGHV7-IGKV16",138),("IGHV7-IGKV2",139),("IGHV7-IGKV22",140),("IGHV7-IGKV3",141),("IGHV7-IGKV4",142),("IGHV7-IGKV6",143),("IGHV7-IGKV8",144),("IGHV7-IGLV1",145),("IGHV7-IGLV2",146),("IGHV7-IGLV3",147),("IGHV7-IGLV6",148),("IGHV7-IGLV7",149),("IGHV7-unknown",150),("IGHV8-IGKV1",151),("IGHV8-IGKV10",152),("IGHV8-IGKV12",153),("IGHV8-IGKV13",154),("IGHV8-IGKV16",155),("IGHV8-IGKV3",156),("IGHV8-IGKV4",157),("IGHV8-IGKV5",158),("IGHV8-IGKV6",159),("IGHV8-IGKV8",160),("IGHV8-IGLV1",161),("IGHV9-IGKV1",162),("IGHV9-IGKV10",163),("IGHV9-IGKV12",164),("IGHV9-IGKV13",165),("IGHV9-IGKV19",166),("IGHV9-IGKV2",167),("IGHV9-IGKV3",168),("IGHV9-IGKV4",169),("IGHV9-IGKV5",170),("IGHV9-IGKV6",171),("IGHV9-IGKV8",172),("IGHV9-IGKV9",173),("IGHV9-IGLV1",174),("IGHV9-IGLV3",175),("IGHV10-IGKV1",176),("IGHV10-IGKV10",177),("IGHV10-IGKV12",178),("IGHV10-IGKV2",179),("IGHV10-IGKV3",180),("IGHV10-IGKV4",181),("IGHV10-IGKV6",182),("IGHV10-IGKV8",183),("IGHV12-IGKV3",184),("IGHV13-IGKV4",185),("IGHV14-IGKV1",186),("IGHV14-IGKV10",187),("IGHV14-IGKV12",188),("IGHV14-IGKV13",189),("IGHV14-IGKV14",190),("IGHV14-IGKV15",191),("IGHV14-IGKV17",192),("IGHV14-IGKV19",193),("IGHV14-IGKV2",194),("IGHV14-IGKV3",195),("IGHV14-IGKV4",196),("IGHV14-IGKV6",197),("IGHV14-IGKV8",198),("IGHV14-IGKV9",199),("unknown-IGKV1",200),("unknown-IGKV2",201),("unknown-IGKV3",202),("unknown-IGKV4",203),("unknown-IGLV1",204),("unknown-IGLV2",205),("unknown-IGLV3",206),("unknown-unknown",207)])

# structure property classes
SS3_class = OrderedDict([('H',0),('E',1),('C',2),('-',-1)])
SS8_class = OrderedDict([('G',0),('I',1),('H',2),('B',3),('E',4),('T',5),('S',6),('-',-1)])
RSA2_class = OrderedDict([('B',0),('E',1),('-',-1)])


class BaseTokenizer():
    """Basic Tokenizer. Can use different vocabs depending on the model.
    """

    def __init__(self, vocab: str = 'pfam'):
        if vocab == 'iupac':
            self.vocab = IUPAC_VOCAB
        elif vocab == 'unirep':
            self.vocab = UNIREP_VOCAB
        elif vocab == 'pfam':
            self.vocab = PFAM_VOCAB
        else:
            raise Exception("vocab not known!")
        self.tokens = list(self.vocab.keys())
        self._vocab_type = vocab
        assert self.start_token in self.vocab and self.stop_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def stop_token(self) -> str:
        return "<sep>"

    @property
    def mask_token(self) -> str:
        if "<mask>" in self.vocab:
            return "<mask>"
        else:
            raise RuntimeError(f"{self._vocab_type} vocab does not support masking")

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_token(id_) for id_ in indices]

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token

    def add_placeholder_tokens(self, token_ids: List[str], placeholder: str):
        """In order to be consistent with amino acid seq tokenization,
        add placeholder token '-' at start and end position for structure element seq
        """
        return [placeholder] + token_ids + [placeholder]

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        tokens = self.add_special_tokens(tokens)
        token_ids = self.convert_tokens_to_ids(tokens)
        return np.array(token_ids, np.int64)
    
    def get_normal_token_ids(self) -> List[int]:
        ids2return = []
        for tok, idx in self.vocab.items():
            if '<' not in tok:
                ids2return.append(idx)
        return ids2return

    def get_normal_token_ids(self) -> List[int]:
        ids2return = []
        for tok, idx in self.vocab.items():
            if '<' not in tok:
                ids2return.append(idx)
        return ids2return

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()


    def struct_convert_tokens_to_ids(self, tokens: List[str], type: str) -> List[int]:
        if type == 'ss3':
            token_dict = SS3_class
        elif type == 'ss8':
            token_dict = SS8_class
        elif type == 'rsa2':
            token_dict = RSA2_class
        else:
            raise Exception(f"invalid input type {type}")
        return [token_dict[token] for token in tokens]

