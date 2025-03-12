import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin #для гридсерча 


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    right_indexes = np.argsort(list((feature_vector)))
    
    sorted_feature_vector = feature_vector[right_indexes]
    
    
    sorted_target_vector = target_vector[right_indexes]
    
    prom_vector = np.unique(sorted_feature_vector)[1:]
    
    thresholds = ((np.unique(sorted_feature_vector)[:-1]) + (prom_vector)) / 2
    
    R = len(target_vector) #количество объектов в текущей вершине 
    
    Rl = np.unique(sorted_feature_vector, return_index=True)[1][1:] #находим индексы уникальных элементов, они и будут показывать, сколько объектов ушли в левого потомка, за исключением первого, так как 0 объектов не может уйти в потомка 
    # P.S эту штучку взял со sTaCk OvErFlOw
    
    Rr = R - Rl # тогда в правой вершине окажутся все, кроме тех, кто попал в левую 
    
    ps_left = np.cumsum(sorted_target_vector)[Rl - 1] #отнимаем единичку, чтобы сдвинуть и правильно посчитать куммулятивные суммы т.е. количество единичек слева для каждого разбиения 
    
    ps_right = np.count_nonzero(sorted_target_vector) - ps_left #количество единичек справа это все единички минус единички которые окажутся слева 
    
    Hl = 1 - (ps_left / Rl) ** 2 - (1 - ps_left / Rl) ** 2 # джини для левых разбиений 
    Hr = 1 - (ps_right / Rr) ** 2 - (1 - ps_right / Rr) ** 2 # джини для правых 
    
    ginis = - (Rl / R) * Hl - (Rr) / R * Hr #вектор оценки качества предикатов по критерию джини 
    
    best = np.argmax(ginis) 
    
    threshold_best = thresholds[best]
    
    gini_best = ginis[best]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))): #если у нас есть не вещественные и не категориальные то мы хз че это поэтому вальюэррор
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf


    def _fit_node(self, sub_X, sub_y, node, depth):
        if (self._max_depth is not None) and (depth == self._max_depth): #выполнен критерий останова для максимальной глубины
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #тогда ответ самый частый класс 
            return 
        
        if (self._min_samples_split is not None) and (len(sub_y) < self._min_samples_split): #выполнен критерий останова для минимального количества объектов в вершине 
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #тогда ответ самый частый класс 
            return
        
        
        if np.all(sub_y == sub_y[0]): # 0 1 0 vs 0 0 0 -> T F T -> T   |   0 0 0 vs 0 0 0 -> T T T -> T     !1-Я ОШИБКА, меняем != на == !
            node["type"] = "terminal"
            node["class"] = sub_y[0] #ответ - самый частый класс, он тут один поэтому все ок 
            return #выполнен критерий останова, возвращаемся 
        

        feature_best, threshold_best, gini_best, split = None, None, None, None
        
        for feature in range(sub_X.shape[1]): #итерируемся по всем признакам, почему-то тут с единицы начинается  !2-Я ОШИБКА, range(sub_X.shape[1])
            feature_type = self._feature_types[feature] #получаем тип признака 
            categories_map = {}

            if feature_type == "real": #если вещественный 
                feature_vector = sub_X[:, feature] #получили вектор признака вещественный  
            elif feature_type == "categorical": #если категориальный 
                counts = Counter(sub_X[:, feature]) #получаем словарик вида {значение признака: количество объектов такого признака} то есть |R_m(u)| 
                clicks = Counter(sub_X[sub_y == 1, feature]) # те объекты ответ на которых положительный и они принадлежат R_m(u)
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks: #если у текущего значения признака есть хотя бы один положительный ответ 
                        current_click = clicks[key] #сохраняем 
                    else:
                        current_click = 0 # если нет то очев 0 ставим 
                    ratio[key] = current_click / current_count #тут написан бред, как минимум мы тут делим на ноль, как максимум мы делим количество кликов на мощность множества объектов из R_m(u) -> !3-Я ОШИБКА, дробь наоборот!
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # окей мы отсортили по доле (x[1]) в sorted верно, но когда мы применяем map мы должны получить признаки а не доли, поэтому x[1] меняем на x[0] -> 4-Я ОШИБКА
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories))))) #перенумеруем отсортированные категории?? и каждому по очереди поставим в соответствие 0, 1, 2 и т.д. чтобы потом Джини их разбил как вещественные 
                
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) #перенумеровываем для текущего признака каждый из категориальных, тут надо обернуть map, потому что иначе получим чет такое array(<map object at 0x2a3592b00>, dtype=object), поэтому фигачим list перед ним -> 5-Я ОШИБКА 
            else:
                raise ValueError #если ни вещественный ни категориальный то хз что это ловвим вэлью эррор 

            if len(np.unique(feature_vector)) == 1: #???? зачем это (UPD: короче если у нас все признаки одинаковые то мы не можем их разбить по джини а значит идем дальше по циклу) -> 6-Я ОШИБКА 
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best: #если пока не было лучшего джини или текущий лучше предыдущего лучшего 
                feature_best = feature #перезаписываем лучший признак 
                gini_best = gini #перезаписываем лучший джини 
                split = feature_vector < threshold #левый сплит получаем массив True и False в текущем векторе значений признака True для тех кто не прошел порог и идут влево   

                if feature_type == "real":
                    threshold_best = threshold #если вещественный признак то лучший порог это найденный джини порог 
                elif feature_type == "categorical": #тут вообще мем, categorical надо с маленькой написать -> 7-Я ОШИБКА
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items()))) #если категориальный, то мы не можем выбрать численный порог, поэтому собираем множество категорий, которые меньше порога 
                else:
                    raise ValueError #если ни вещественный ни категориальный то снова ловим вэльюэррор

        if feature_best is None: #если не нашли лучший признак для разбиения значит попали в лист, то есть выполнен второй критерий останова 
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # один самый частый класс в sub_y, тут синтаксис, сначала обратимся к первой паре в спике а потом к первому элементу то есть 8-Я ОШИБКА 
            return #конец бабе капе 
        
        

        node["type"] = "nonterminal" #если не прокатили критерии останова то мы точно не в листе 

        node["feature_split"] = feature_best #запоминаем лучший признак для разбиения 
        if self._feature_types[feature_best] == "real": 
            node["threshold"] = threshold_best 
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError #тут все очев 
        
        how_many_left = np.count_nonzero(split)
        
        if (self._min_samples_leaf is not None) and ((how_many_left < self._min_samples_leaf) or (len(sub_y) - how_many_left < self._min_samples_leaf)): #критерий останова: минимальное количество объектов в листе 
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] #ответ - самый частый класс 
            return
        
        
        
        
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth+1) #сплит это у нас чуваки прошедшие в левый лист
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth+1)  #тут для sub_X все ок, а для sub_y почему-то вправо отправляем левых чуваков -> 9-Я ОШИБКА np.logical_not(split)


    def _predict_node(self, x, node): #x здесь это объект 
        if node['type'] == 'terminal': #если текущая вершина листовая то возвращаем предсказанный класс 
            return node["class"]
        else:
            if self._feature_types[node['feature_split']] == 'real': #если лучший признак для разбиения вершины вещественный
                if x[node['feature_split']] < node['threshold']: #если значение признака для нашего объекта меньше порога то пусть левый решает че с ним делать 
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"]) #иначе в правый кидаем 
                
            elif self._feature_types[node['feature_split']] == 'categorical':#если лучший признак для разбиения вершины категориальный 
                if x[node['feature_split']] in node["categories_split"]: #мы запомнили признаки, которые по разбиению пошли влево поэтому если он там есть то кидаем его влево 
                    return self._predict_node(x, node["left_child"])
                else: #иначе вправо 
                    return self._predict_node(x, node["right_child"]) 


    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)
        

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    
class LinearRegressionTree(BaseEstimator, RegressorMixin): #наследуемся для грисерча 
    def __init__(self, feature_types, base_model_type=LinearRegression, max_depth=None, min_samples_split=None, min_samples_leaf=None, quantiles=10): # по дефолту обычная линейная регрессия + добавим новый параметр - количество квантилей, по дефолту поставим 10 потому что так все сделали лол 
        if np.any(list(map(lambda x: x != "real", feature_types))): #забиваем смачный болт на категориальные 
            raise ValueError("There is unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._base_model_type = base_model_type #модель линрега 
        self._quantiles = quantiles #новая перменная количество квантилей для разбиения 
        
    def _get_model(self, X, y): # функция, которая вернет нам зафичинную модель  
        model = self._base_model_type().fit(X, y)
        return model
    
    def _get_loss(self, X, y, model): #функция возвращающая ошибку учимся на мсе как бы да 
        y_pred = model.predict(X)
        return mean_squared_error(y, y_pred)
    
    
    def _get_thresholds(self, feature_vector): #возращает пороги по квантилям 
        feature_vector = np.unique(feature_vector) #хотим только для уникальных значений вектора признаков считать 
        if len(feature_vector) == 1: #если константный то ниче не подобрали 
            return []
        qs = np.linspace(0, 1, self._quantiles+2)[1:-1] #равномерно распределяем квантили и обрубаем нулевой и единичный чтобы не допустить пустого разбиения 
        return np.unique(np.quantile(feature_vector, qs)) #если вдруг квантили получились одинаковые то берем только уникальные 

    
    def _find_best_split_linregtree(self, feature_vector, y, X):
        best_loss = self._get_loss(X, y, self._get_model(X, y)) #текущий loss 
        best_threshold = None  
        thresholds = self._get_thresholds(feature_vector) #получаем пороги по квантилям 
        for threshold in thresholds: #итерируемся по порогам 
            nall = len(y) #количество элементов в вершине 
            lefts = feature_vector < threshold #маска для левых 
            rights = np.logical_not(lefts) #маска для правых 
            nlefts = np.count_nonzero(lefts) #количество левых 
            nrights = nall - nlefts #количество правых 
            X_left, y_left = X[lefts], y[lefts] #кого отправим влево 
            X_right, y_right = X[rights], y[rights] #кого отправим вправо 
            loss = nlefts / nall * (self._get_loss(X_left, y_left, self._get_model(X_left, y_left))) + nrights / nall * (self._get_loss(X_right, y_right, self._get_model(X_right, y_right))) #наша метрика качества предиката
            if loss < best_loss: #если лучше чем бест то сохраняем 
                best_loss = loss
                best_threshold = threshold
        return best_threshold, best_loss #возвращаем лучший порог и лосс 
            
    
    
    def _fit_node(self, sub_X, sub_y, node, depth):
        if (self._max_depth is not None) and (depth == self._max_depth): # критерий останова - максимальная глубина выполнен 
            node['type'] = 'terminal'
            node['model'] = self._get_model(sub_X, sub_y) #в листе - модель 
            return 
        
        if (self._min_samples_split is not None) and (len(sub_y) < self._min_samples_split): # выполнен критерий останова для минимального количества объектов в вершине 
            node["type"] = "terminal"
            node["model"] = self._get_model(sub_X, sub_y) # в листе - модель 
            return
        
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["model"] = self._get_model(sub_X, sub_y)
            return 
        
        feature_best, threshold_best, loss_best, split = None, None, None, None
        
        for feature in range(sub_X.shape[1]): #итерируемся по всем признакам
            feature_type = self._feature_types[feature] #получаем тип признака 
            
            if feature_type == "real": #если вещественный 
                feature_vector = sub_X[:, feature] #получили вектор признака вещественный  
            else:
                raise ValueError #если не вещественный, то ловим вэлью эррор 
            
    
            if len(np.unique(feature_vector)) == 1: #проверка на константный признак 
                continue
            
            threshold, loss = self._find_best_split_linregtree(feature_vector, sub_y, sub_X)
            
            
            if loss_best is None or loss < loss_best: #если пока не было лучшего лосса или текущий лучше предыдущего лучшего 
                feature_best = feature #перезаписываем лучший признак 
                loss_best = loss #перезаписываем лучший лосаа 
                split = feature_vector < threshold #левый сплит получаем массив True и False в текущем векторе значений признака True для тех кто не прошел порог и идут влево   

                if feature_type == "real":
                    threshold_best = threshold #если вещественный признак то лучший порог это найденный лосс порог 
                else:
                    raise ValueError 

        if feature_best is None: #если не нашли лучший признак для разбиения значит попали в лист, то есть выполнен второй критерий останова 
            node["type"] = "terminal"
            node["model"] = self._get_model(sub_X, sub_y) 
            return #конец бабе капе 
        
        
        node["type"] = "nonterminal" #если не прокатили критерии останова то мы точно не в листе 

        node["feature_split"] = feature_best #запоминаем лучший признак для разбиения 
        if self._feature_types[feature_best] == "real": 
            node["threshold"] = threshold_best 
        else:
            raise ValueError #тут все очев 
        
        how_many_left = np.count_nonzero(split)
        
        if (self._min_samples_leaf is not None) and ((how_many_left < self._min_samples_leaf) or (len(sub_y) - how_many_left < self._min_samples_leaf)): #критерий останова: минимальное количество объектов в листе 
            node["type"] = "terminal"
            node["model"] =  self._get_model(sub_X, sub_y) #ответ - модель 
            return
        
        
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth+1) #магия рекурсии, строим левое и правое поддеревья
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth+1) 
        
        
    def _predict_node(self, x, node): #функция предикт 
        if node['type'] == 'terminal': #если в листе 
            return node["model"].predict(x.reshape(1, -1))[0] #решейпим x, чтобы он нормально запредиктился и предиктим 
        else:
            if x[node['feature_split']] < node['threshold']: #если не прошли порог то пусть левый разбирается 
                return self._predict_node(x, node["left_child"]) 
            else:
                return self._predict_node(x, node["right_child"]) #иначе правый 
        
    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)
    
    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    
    def get_params(self, deep=True): #это мне делал GPT потому что я в душе не чаю как такие штуки делать для гридсерча 
        return {
            'feature_types': self._feature_types,
            'base_model_type': self._base_model_type,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf,
            'quantiles': self._quantiles
        }
    
    def set_params(self, **params): #аналогично 
        for key, value in params.items():
            if key == 'feature_types':
                self._feature_types = value
            elif key == 'base_model_type':
                self._base_model_type = value
            elif key == 'max_depth':
                self._max_depth = value
            elif key == 'min_samples_split':
                self._min_samples_split = value
            elif key == 'min_samples_leaf':
                self._min_samples_leaf = value
            elif key == 'quantiles':
                self._quantiles = value
            else:
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}.")
        return self

    
