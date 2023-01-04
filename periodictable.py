
rcovs = {'H': 0.699198669167688, 'He': 0.6047123625234059, 'Li': 2.532233018066762, 'Be': 1.7007535195970789, 'B': 1.5495754289662274,
            'C': 1.4550891223219453, 'N': 1.4172945996642325, 'O': 1.3795000770065196, 'F': 1.3417055543488066, 'Ne': 1.3039110316910938,
            'Na': 2.9101782446438906, 'Mg': 2.4566439727513365, 'Al': 2.229876836805059, 'Si': 2.097596007503064, 'P': 2.003109700858782,
            'S': 1.9275206555433562, 'Cl': 1.870828871556787, 'Ar': 1.833034348899074, 'K': 3.703863220455861, 'Ca': 3.2881234712210192,
            'Sc': 2.721205631355326, 'Ti': 2.570027540724475, 'V': 2.3621576661070542, 'Cr': 2.399952188764767, 'Mn': 2.626719324711044,
            'Fe': 2.3621576661070542, 'Co': 2.3810549274359105, 'Ni': 2.2865686207916283, 'Cu': 2.6078220633821876, 'Zn': 2.4755412340801928,
            'Ga': 2.3810549274359105, 'Ge': 2.3054658821204845, 'As': 2.2487740981339153, 'Se': 2.192082314147346, 'Br': 2.154287791489633,
            'Kr': 2.078698746174208, 'Rb': 3.987322140388707, 'Sr': 3.628274175140435, 'Y': 3.061356335274742, 'Zr': 2.796794676670752,
            'Nb': 2.5889248020533313, 'Mo': 2.740102892684183, 'Tc': 2.9479727673016036, 'Ru': 2.3810549274359105, 'Rh': 2.5511302793956188,
            'Pd': 2.4755412340801928, 'Ag': 2.8912809833150344, 'Cd': 2.796794676670752, 'In': 2.721205631355326, 'Sn': 2.664513847368757,
            'Sb': 2.6078220633821876, 'Te': 2.5511302793956188, 'I': 2.5133357567379058, 'Xe': 2.4566439727513365, 'Cs': 4.251883798992697,
            'Ba': 3.741657743113574, 'Lu': 3.023561812617029, 'Hf': 2.834589199328465, 'Ta': 2.6078220633821876, 'W': 2.759000154013039,
            'Re': 3.004664551288173, 'Os': 2.4188494500936235, 'Ir': 2.5889248020533313, 'Pt': 2.4188494500936235, 'Au': 2.721205631355326,
            'Hg': 2.8156919379996084, 'Tl': 2.796794676670752, 'Pb': 2.7778974153418954, 'Bi': 2.759000154013039, 'Rn': 2.740102892684183, }

elementSymbolToNumber = {'H': 1,'He': 2,'Li': 3,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'Ne': 10,
                        'Na': 11,'Mg': 12,'Al': 13,'Si': 14,'P': 15,'S': 16,'Cl': 17,'Ar': 18,'K': 19,'Ca': 20,
                        'Sc': 21,'Ti': 22,'V': 23,'Cr': 24,'Mn': 25,'Fe': 26,'Co': 27,'Ni': 28,'Cu': 29,'Zn': 30,
                        'Ga': 31,'Ge': 32,'As': 33,'Se': 34,'Br': 35,'Kr': 36,'Rb': 37,'Sr': 38,'Y': 39,'Zr': 40,
                        'Nb': 41,'Mo': 42,'Tc': 43,'Ru': 44,'Rh': 45,'Pd': 46,'Ag': 47,'Cd': 48,'In': 49,'Sn': 50,
                        'Sb': 51,'Te': 52,'I': 53,'Xe': 54,'Cs': 55,'Ba': 56,'La': 57,'Ce': 58,'Pr': 59,'Nd': 60,
                        'Pm': 61,'Sm': 62,'Eu': 63,'Gd': 64,'Tb': 65,'Dy': 66,'Ho': 67,'Er': 68,'Tm': 69,'Yb': 70,
                        'Lu': 71,'Hf': 72,'Ta': 73,'W': 74,'Re': 75,'Os': 76,'Ir': 77,'Pt': 78,'Au': 79,'Hg': 80,
                        'Tl': 81,'Pb': 82,'Bi': 83,'Po': 84,'At': 85,'Rn': 86,'Fr': 87,'Ra': 88,'Ac': 89,'Th': 90,
                        'Pa': 91,'U': 92,'Np': 93,'Pu': 94,'Am': 95,'Cm': 96,'Bk': 97,'Cf': 98,'Es': 99,'Fm': 100,
                        'Md': 101,'No': 102,'Lr': 103,'Rf': 104,'Db': 105,'Sg': 106,'Bh': 107,'Hs': 108,'Mt': 109,'Ds': 110,
                        'Rg': 111,'Cn': 112,'Nh': 113,'Fl': 114,'Mc': 115,'Lv': 116,'Ts': 117,'Og': 118}

elementSymbols = [' ', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
        'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
        'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
        'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
        'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

valenceElectronsDict = {
'H': 1, 'He': 2,'Li': 1,'Be': 2,'B': 3,'C': 4,'N': 5,'O': 6,'F': 7,'Ne': 8,'Na': 1,'Mg': 2,'Al': 3,'Si': 4,'P': 5,'S': 6,
'Cl': 7,'Ar': 8,'K': 1,'Ca': 2,'Sc': 2,'Ti': 2,'V': 2,'Cr': 1,'Mn': 2,'Fe': 2,'Co': 2,'Ni': 2,'Cu': 1,'Zn': 2,'Ga': 3,'Ge': 4,
'As': 5,'Se': 6,'Br': 7,'Kr': 8,'Rb': 1,'Sr': 2,'Y': 2,'Zr': 2,'Nb': 1,'Mo': 1,'Tc': 2,'Ru': 1,'Rh': 1,'Pd': 18,'Ag': 1,'Cd': 2,
'In': 3,'Sn': 4,'Sb': 5,'Te': 6,'I': 7,'Xe': 8,'Cs': 1,'Ba': 2,'La': 2,'Ce': 2,'Pr': 2,'Nd': 2,'Pm': 2,'Sm': 2,'Eu': 2,'Gd': 2,'Tb': 2,
'Dy': 2,'Ho': 2,'Er': 2,'Tm': 2,'Yb': 2,'Lu': 2,'Hf': 2,'Ta': 2,'W': 2,'Re': 2,'Os': 2,'Ir': 2,'Pt': 1,'Au': 1,'Hg': 2,'Tl': 3,'Pb': 4,
'Bi': 5,'Po': 6,'At': 7,'Rn': 8,'Fr': 1,'Ra': 2,'Ac': 2,'Th': 2,'Pa': 2,'U': 2,'Np': 2,'Pu': 2,'Am': 2,'Cm': 2,'Bk': 2,'Cf': 2,'Es': 2,
'Fm': 2,'Md': 2,'No': 2,'Lr': 2,'Rf': 2,'Db': 2,'Sg': 2,'Bh': 2,'Hs': 2,'Mt': 2,'Ds': 1,'Rg': 1,'Cn': 2,'Nh': 3,'Fl': 4,'Mc': 5,'Lv': 6,
'Ts': 7,'Og': 8
}


def getRcov_n(el):
    return rcovs[elementSymbols[el]]

def getCovalentRadiosFromString(element):
    return getRcov_n(elementSymbolToNumber[element])

def get_rcov_dict():
    return rcovs


def getNumberFromElementSymbol(elemnt_symbol: str):
    return elementSymbolToNumber[elemnt_symbol]


def getCoreAndValenceElectrons(elementNumber):
    if type(elementNumber) == int:
        symb = elementSymbols[elementNumber]
    elif type(elementNumber) == str:
        symb = elementNumber
    valenceElectrons = valenceElectronsDict[symb]
    coreElectrons = elementSymbolToNumber[symb] - valenceElectrons
    return coreElectrons, valenceElectrons


if __name__ == '__main__':
    print(getCoreAndValenceElectrons('C'))
    for i in range(1, 19):
        print(getCoreAndValenceElectrons(i))


