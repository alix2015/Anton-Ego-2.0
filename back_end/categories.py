import json


class Categories(object):
    def __init__(self, in_file=None, out_file=None):
        self.dic = {}
        self._initialize()
        if in_file:
            with open(in_file, 'r') as f:
                json_str = f.read()
                self.dic = json.loads(json_str)
            f.close()
        if out_file:
            self._tojson(self.dic, out_file)

    def __len__(self):
        return len(self.dic)

    def __contains__(self, key):
        return (key in self.dic)

    def __iter__(self):
        return self.dic.__iter__()

    def _initialize(self):
        self.dic['food'] = {0, 6, 8, 9, 10, 13, 17, 18, 22, 23, 25, 26, 27, 28,
                            29, 32, 34, 36, 37, 38, 40, 42, 43, 44, 45, 47, 49,
                            50, 52, 56, 57, 58, 63, 65, 66, 67, 70, 74, 76, 77,
                            78, 80, 81, 99}
        self.dic['service'] = {1, 14, 15, 21, 31, 46, 86}
        self.dic['ambience'] = {3, 4, 5, 18, 19, 35, 39, 48, 59, 61, 68, 72, 82,
                                84, 90}

        self.dic['price'] = {7, 16, 31}

        self.dic['wine'] = {0, 9, 17, 26, 38}
        self.dic['cocktail'] = {28, 58}
        self.dic['beer'] = {66}
        self.dic['wait'] = {1, 86}
        self.dic['special occasion'] = {2, 33, 35, 48, 60}
        self.dic['noise'] = {3}
        self.dic['decor'] = {4, 18, 59}
        self.dic['parking'] = {5}
        self.dic['location'] = {5, 94}
        self.dic['salad'] = {6, 56}
        self.dic['meat'] = {8, 27, 29, 57}
        self.dic['entree'] = {8, 10, 18, 25, 27, 29, 32, 34, 37, 40, 42, 43,
                              44, 45, 49, 52, 57, 70, 74, 78}
        self.dic['side'] = {67}
        self.dic['vegetarian'] = {9, 34}
        self.dic['seafood'] = {10, 18, 37, 43, 74}
        self.dic['dessert'] = {13}
        self.dic['reservation'] = {14, 83}
        self.dic['classic'] = {19}
        self.dic['slow'] = {21}
        self.dic['presentation'] = {23, 36}
        self.dic['surprise'] = {53}
   

    def _tojson(self, dictionary, filename):
        with open(filename, 'wb') as f:
            json.dump(dictionary, f)
        f.close()

    def get(self, category):
        return self.dic.get(category, {})


if __name__ == '__main__':
    out_file = '../front_end/data/categories_2a_extraSW.json'
    cat = Categories(out_file=out_file)
