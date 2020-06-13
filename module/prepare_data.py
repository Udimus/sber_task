"""
Tools to load and preprocess data
"""
import logging

from sklearn.preprocessing import LabelBinarizer
from sklearn.base import TransformerMixin
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
CAT_FEATURES = [
    'feat_1',
    'feat_2',
    'feat_4',
    'feat_11',
    'feat_15',
    'feat_20',
    'feat_27',
    'feat_30',
    'feat_38',
    'feat_39',
    'feat_43',
    'feat_48',
    'feat_50',
    'feat_53',
    'feat_54',
    'feat_57',
    'feat_58',
    'feat_61',
    'feat_62',
    'feat_66',
    'feat_67',
    'feat_69',
    'feat_72',
    'feat_73',
    'feat_77',
    'feat_78',
    'feat_81',
    'feat_88',
    'feat_89',
    'feat_92',
    'feat_96',
    'feat_98',
    'feat_103',
    'feat_104',
    'feat_106',
    'feat_112',
    'feat_113',
    'feat_114',
    'feat_115',
    'feat_121',
    'feat_123',
    'feat_126',
    'feat_128',
    'feat_129',
]
NUM_FEATURES = [
    'feat_5',
    'feat_9',
    'feat_12',
    'feat_22',
    'feat_25',
    'feat_37',
    'feat_55',
    'feat_56',
    'feat_59',
    'feat_60',
    'feat_86',
    'feat_95',
    'feat_119',
    'feat_125',
]
BIN_FEATURES = [
    'feat_0',
    'feat_3',
    'feat_6',
    'feat_7',
    'feat_8',
    'feat_10',
    'feat_13',
    'feat_14',
    'feat_16',
    'feat_17',
    'feat_18',
    'feat_19',
    'feat_21',
    'feat_23',
    'feat_24',
    'feat_26',
    'feat_28',
    'feat_29',
    'feat_31',
    'feat_32',
    'feat_33',
    'feat_34',
    'feat_35',
    'feat_36',
    'feat_40',
    'feat_41',
    'feat_42',
    'feat_44',
    'feat_45',
    'feat_46',
    'feat_47',
    'feat_49',
    'feat_51',
    'feat_52',
    'feat_63',
    'feat_64',
    'feat_65',
    'feat_68',
    'feat_70',
    'feat_71',
    'feat_74',
    'feat_75',
    'feat_76',
    'feat_79',
    'feat_80',
    'feat_82',
    'feat_83',
    'feat_84',
    'feat_85',
    'feat_87',
    'feat_90',
    'feat_91',
    'feat_93',
    'feat_94',
    'feat_97',
    'feat_99',
    'feat_100',
    'feat_101',
    'feat_102',
    'feat_105',
    'feat_107',
    'feat_108',
    'feat_109',
    'feat_110',
    'feat_111',
    'feat_116',
    'feat_117',
    'feat_118',
    'feat_120',
    'feat_122',
    'feat_124',
    'feat_127',
]
EXP = 0.00001
RANDOM_SEED = 71


class BinFeaturesTransformer(TransformerMixin):
    """
    Preprocess binary features as categorical or numeric
    """
    def __init__(
            self,
            num_features=None,
            cat_features=None,
            bin_features=None,
            bin_as_numeric=False,
    ):
        if num_features is None:
            num_features = NUM_FEATURES
        if cat_features is None:
            cat_features = CAT_FEATURES
        if bin_features is None:
            bin_features = BIN_FEATURES
        self._cat_features = cat_features
        self._num_features = num_features
        self._bin_features = bin_features
        self._bin_as_numeric = bin_as_numeric
        self._encoders = dict()

    def fit(self, df, *args):
        logger.debug('Fitting Bin transformer...')
        if self._bin_as_numeric:
            for col in self._bin_features:
                encoder = LabelBinarizer()
                encoder.fit(df[col])
                self._encoders[col] = encoder
        return self

    def transform(self, df):
        transformed_df = df[
            self._cat_features
            + self._num_features
            + self._bin_features
        ].copy()
        logger.debug('Applying Bin transformer...')
        if self._bin_as_numeric:
            for col in self._bin_features:
                encoder = self._encoders[col]
                transformed_df[col] = encoder.transform(transformed_df[col])
        return transformed_df

    def get_features(self):
        if self._bin_as_numeric:
            return {
                'num_features': self._num_features + self._bin_features,
                'cat_features': self._cat_features,
            }

        return {
            'num_features': self._num_features,
            'cat_features': self._cat_features + self._bin_features,
        }


class DropTransformer(TransformerMixin):
    """
    Drop selected columns from dataset
    """
    def __init__(self, drop_columns,):
        self.drop_columns = drop_columns

    def fit(self, df, *args):
        return self

    def transform(self, df):
        return df.drop(columns=self.drop_columns)


class CatFeaturesTransformer(TransformerMixin):
    """
    Convert categorical features to numeric using target.
    """
    def __init__(
            self,
            stat_type='mean',
            expanding=False,
            folds_number=5,
            alpha=0,
            cat_features=None,
    ):
        """
        :param stat_type: statistic type to replace categorical value.
            Now only 'mean' is implemented.
        :param expanding:
            If True, we place at each place in train dataset
            statistic of target from previous n-1 rows.
            If False, for train we use k-folds for train dataset,
            for test we use full train statistic.
        :param folds_number:
            Number of folds. Only if expanding is False.
        :param alpha:
            Smoothing coefficient.
            For example, we want to replace 'value' for 'cat'.
            If sub_df = df[df['cat']=='value'], n_rows = len(sub_df),
            than
            'cat_value' -> (stat(sub_df[target]) * n_rows
                            + stat(df) * alpha) / (n_rows + alpha)
            alpha==0 means no smoothing.
        :param cat_features:
            List of columns with categorical features to replace.
        :param target: Name of target column.
        """
        self._alpha = alpha
        self._folds_number = folds_number
        self._stat_type = stat_type
        self._use_expanding = expanding
        self._encodings = dict()
        self._global_mean = None
        if cat_features is None:
            cat_features = CAT_FEATURES
        self._cat_features = cat_features
        self._target = 'target'

    def _get_grouped_means(self, df, col):
        cat_sum = df.groupby(col)[self._target].sum()
        cat_count = df.groupby(col)[self._target].count()
        return ((cat_sum + self._global_mean * self._alpha)
                / (cat_count + self._alpha)).to_dict()

    def _fill_with_stats(self, df, col, cat_stats):
        return (
            df[col]
            .map(cat_stats)
            .fillna(self._global_mean)
            .values
        )

    def fit(self, df, target):
        df = df.copy()
        df[self._target] = target
        logger.debug('Fitting Cat Transformer...')
        self._global_mean = df[self._target].mean()
        logger.debug('global mean: %.4f', self._global_mean)
        for col in self._cat_features:
            self._encodings[col] = self._get_grouped_means(df, col)

    def transform(self, df):
        logger.debug('Transforming test dataset...')
        df = df.copy()
        for col in self._encodings:
            df[col] = self._fill_with_stats(df, col, self._encodings[col])
        return df

    def _expanding_transform(self, df, col):
        cumsum = (df.groupby(col)[self._target].cumsum()
                  - df[self._target])
        cumcount = df.groupby(col).cumcount()
        return ((cumsum + self._global_mean * self._alpha)
                / (cumcount + self._alpha)).fillna(0)

    def _kfold_transform(self, df, col):
        splitter = StratifiedKFold(
            n_splits=self._folds_number,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
        filler = pd.Series(index=df.index, dtype=np.float64)
        for remaining_idx, batch_idx in splitter.split(df, df[col]):
            batch = df.iloc[batch_idx]
            remaining_df = df.iloc[remaining_idx]
            cat_means = self._get_grouped_means(remaining_df, col)
            filler.iloc[batch_idx] = self._fill_with_stats(
                batch, col, cat_means)
        return filler

    def fit_transform(self, df, target):
        self.fit(df, target)
        df = df.copy()
        df[self._target] = target
        logger.debug('Transforming train dataset...')
        for col in self._cat_features:
            if self._use_expanding:
                df[col] = self._expanding_transform(df, col)
            else:
                df[col] = self._kfold_transform(df, col)
        return df.drop(columns=[self._target], inplace=False)
