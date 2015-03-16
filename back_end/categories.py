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
        self.dic['Food'] = {7, 10, 20, 24, 25, 27, 30, 33, 36, 37, 42, 43,
                            44, 47, 49, 54, 55, 58, 62, 68, 73, 82, 88, 91}
        self.dic['Service'] = {0, 1, 2, 13, 15, 29, 34, 40, 48, 57, 61,
                               66, 67, 70, 89}
        self.dic['Ambience'] = {4, 5, 13, 32, 41, 50, 57, 67, 72, 85, 87}

        self.dic['Price'] = {8, 51, 78, 99}
        self.dic['Reservation'] = {11, 35, 92}
        self.dic['Location'] = {31, 65, 85}

        self.dic['Celebration'] = {3, 18, 41, 56, 63}
        self.dic['Experience'] = {4, 5, 79, 84, 86}
        self.dic['Recommend'] = {12, 46, 53}
        self.dic['Wait'] = {14}
        self.dic['Lunch'] = {26}
        self.dic['Expectation'] = {28}
        self.dic['Noise'] = {32}
        self.dic['History'] = {39, 64}
        self.dic['Seating'] = {87}

        self.dic['Meat'] = {36, 44, 49}
        self.dic['Brunch'] = {37}
        self.dic['French'] = {43}
        self.dic['Comfort'] = {7}
        self.dic['Menu'] = {10, 73}
        self.dic['Salad'] = {20}
        self.dic['Drinks'] = {47}
        self.dic['Dessert'] = {88}
   

    def _tojson(self, dictionary, filename):
        with open(filename, 'wb') as f:
            json.dump(dictionary, f)
        f.close()

    def get(self, category):
        '''
        INPUT: Categories object, string
        OUTPUT: set of integers

        This method returns the index of the latent features composing
        category.
        '''
        return self.dic.get(category, {})


if __name__ == '__main__':
    out_file = '../front_end/data/categories_2a_extraSW.json'
    cat = Categories(out_file=out_file)
