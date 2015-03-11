import json


class Categories(object):
    def __init__(self, in=None, out=None):
        self.dic = {}
        if in:
            with open(in, 'r') as f:
                json_str = f.read()
                self.dic = json.loads(json_str)
            f.close()
        else:
            self._initialize()
        if out:
            self._tojson(out)
    
    def _initialize(self):
        self.dic['food'] = {4, 6, 7, 21, 24, 32, 34, 35, 2, 36, 25, 37,
                            38, 39, 40, 41, 42, 44, 47, 48, 53, 54, 55,
                            56, 65, 68, 69, 70, 73, 75, 79, 85, 91, 93,
                            95, 97}
        self.dic['service'] = {5, 9, 15, 18, 19, 45, 58, 60, 62, 66, 71,
                               72, 81, 90, 94, 98}
        self.dic['ambience'] = {10, 16, 20, 26, 31, 61, 64}
        
        self.dic['wine'] = {2, 36}
        self.dic['cocktail'] = {25, 36}
        self.dic['steak'] = {21}
        self.dic['Chinese'] = {24}
        self.dic['French'] = {47, 87}
        self.dic['cheese'] = {34, 97}
        self.dic['dessert'] = {35, 40, 95}
        self.dic['vegetables'] = {37}
        self.dic['meat'] = {42, 44, 47, 49, 51, 53, 69, 70, 85, 91}
        self.dic['pork'] = {51}
        self.dic['steak'] = {49, 53, 69, 70, 85}
        self.dic['egg'] = {44}
        self.dic['potato'] = {44, 48}
        self.dic['entree'] = {38, 39}
        self.dic['layout'] = {16, 26}
        self.dic['noise'] = {17, 64, 82}
        self.dic['music'] = {64, 82}
        self.dic['location'] = {26, 50, 52, 76, 77}
        self.dic['vegetarian'] = {56, 87}
        self.dic['salad'] = {87}
        self.dic['brunch'] = {65, 90}
        self.dic['Mediterranean'] = {73}
        self.dic['Indian'] = {79}
        
        self.dic['excellent'] = {3, 5, 18, 20, 25, 27, 29, 33, 34, 96, 99}
        self.dic['positive sentiment'] = {9, 10, 11, 12, 15, 19, 
                                          22, 28, 45, 46, 54, 59, 60, 62,
                                          63, 66, 68, 80, 81, 86, 90, 94}
        self.dic['negative sentiment'] = {46, 58, 71, 94}
        self.dic['experience'] = {8, 78, 92}
        self.dic['positive recommendation'] = {13, 23, 30, 74, 83}
        self.dic['special occasion'] = {14, 31, 43, 59, 74, 84, 89}
        self.dic['reservation'] = {60}
        self.dic['price'] = {67}
        self.dic['cook'] = {68, 75}

    def _tojson(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.dic, f)
        f.close()

    def get(self, category):
        return self.dic.get(category, {})
