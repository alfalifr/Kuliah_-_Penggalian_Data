from typing import Dict, Callable, Tuple

from ipykernel.pickleutil import istype
from numpy import number
from pandas import Series, DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


def get_res(name: str) -> str: pass


def unique_id_int(data) -> Dict[any, int]:
    """
    Mengambil id unik dari `data`.

    :param data: array like.
    :return: dict[T, int], peta member unik `data` dengan id int nya.
    """
    map_ = {}
    id = 0
    for e in data:
        if e not in map_.keys():
            map_[e] = id
            id += 1
    return map_


def set_unique_id_int(data: Series, autoname_bolean_col: bool = True, return_map: bool = False) -> Series:
    """
    Mengambil `Series` dengan nilai membernya menjadi id unik int.

    :param data: pandas.Series.
    :param return_map: `True` maka fungsi ini akan me-return map uniqeId-nya dari `unique_id_int` juga.
    :param autoname_bolean_col: `True` -> jika kolom tipe `object` dengan nilai id hanya 2, maka nama kolom tersebut
        akan diubah menjadi nilai yang memiliki nilai 1.

    :return: pandas.Series dengan membernya id unik int.
    """
    #print("afhaiuhgphagha")
    if istype(data, DataFrame):
        raise TypeError("`data` merupakan `DataFrame` yang dapat menyebabkan error karena tiap elemen iterasi "
                        "mengembalikan `Series`.")

    ids = unique_id_int(data)
    if autoname_bolean_col and len(ids.keys()) == 2:
        data = data.copy()
        new = None
        for k in ids:
            if ids[k] == 1:
                new = k
                break
        if new is not None:
            data.name = new
    # noinspection PyTypeChecker
    res = data.apply(lambda x: ids[x])
    if not return_map:
        return res
    else:
        return res, ids


def numerize_obj_cols(
        data: DataFrame,
        change_name: Callable[[str], int] = None,
        autoname_bolean_col: bool = True,
        return_map: bool = False,
) -> DataFrame:
    """
    Mengubah kolom `data` yang bertipe `object` menjadi `int`.

    :param data: pandas.DataFrame.
    :param change_name: Fungsi untuk mengubah nama kolom yang bertipe `object`.
    :param autoname_bolean_col: `True` -> jika kolom tipe `object` dengan nilai id hanya 2, maka nama kolom tersebut
        akan diubah menjadi nilai yang memiliki nilai 1.
    :param return_map: `True` maka fungsi ini akan me-return map uniqeId-nya dari `unique_id_int` juga.

    :return: `DataFrame` yang semua kolom bertipe `object` menjadi `int`.
    """
    data2 = data.copy()
    ids_map = {}
    for col in data2.columns:
        col_obj = data2[col]
        print(col, col_obj)
        if col_obj.dtype == "object":
            if return_map:
                new, ids = set_unique_id_int(col_obj, autoname_bolean_col, return_map)
            else:
                new = set_unique_id_int(col_obj, autoname_bolean_col, return_map)
            old = data2[col].name
            data2[col] = new
            data2[col].name = new.name
            data2 = data2.rename(columns={old : new})
            if change_name is not None:
                data2[new].nama = change_name(old)
            if return_map:
                ids_map[data2[new].nama] = ids
    if not return_map:
        return data2
    else:
        return data2, ids_map


def benchmark_train_size(
        method, x, y,
        #metode_name = None, name = None,
        train_size=[0.9, 0.8, 0.75, 0.7, 2/3, 1/2, 0.6, 0.4]
) -> Tuple[int, number]:
    method_name = method.__class__.__name__
    print(f"\n\n ============= Benchmark Train Size - {method_name} ================")

    try: method.fit
    except AttributeError: raise AttributeError("Param `metode` tidak punya fungsi `fit`")

    try: method.score
    except AttributeError: raise AttributeError("Param `metode` tidak punya fungsi `score`")

    best_size = -1
    acc_score = -1
    for t in train_size:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=t, random_state=24)
        method.fit(x_train, y_train)
        score = method.score(x_test, y_test)
        if acc_score < score:
            acc_score = score
            best_size = t
        print("Accuracy %s with train size %.2f : %.3f%%" % (method_name, t, score * 100))
    return best_size, acc_score
