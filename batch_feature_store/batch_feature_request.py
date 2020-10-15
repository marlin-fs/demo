import pyarrow.parquet as pq
import fsspec
from collections import defaultdict
import uuid
from abc import ABC
from abc import abstractmethod
from marlin_service_pb2 import FeatureRequestDetails
import pandas as pd


class BatchFeaturesRequest(ABC):
    @abstractmethod
    def get_batch_features(self, entity_df, features):
        pass


target_timestamp_str = 'target_timestamp'
event_time_str = 'event_time'
uuid_str = 'uuid'


class PandasParquetBatchFeaturesRequest(BatchFeaturesRequest):
    def __init__(self, root_location, marlin_stub):
        self.root_location = root_location
        self.marlin_stub = marlin_stub

    def get_batch_features(self, entity_df, features):
        return PandasParquetBatchFeaturesRequest.get_batch_features_helper(entity_df, features,
                                                                           self.root_location, self.marlin_stub)

    @staticmethod
    def get_batch_features_helper(entity_df, feature_list, root_location, marlin_stub):
        feature_group_dict = defaultdict(lambda: [])
        for feature_def in feature_list:
            split = feature_def.split(':', 1)
            feature_group_dict[split[0]].append(split[1])

        entity_df[uuid_str] = entity_df.apply(lambda _: uuid.uuid4(), axis=1)
        final_df = entity_df.set_index(uuid_str)
        for feature_group, features in feature_group_dict.items():
            feature_group_def = PandasParquetBatchFeaturesRequest.read_feature_group_def(marlin_stub, feature_group)
            feature_entities = list(dict(feature_group_def.entities).keys())
            columns_to_read = feature_entities + features + [event_time_str]
            feature_df = PandasParquetBatchFeaturesRequest.read_files_to_pandas(root_location, feature_group,
                                                                                columns_to_read)
            col_dict = {}
            for col in features:
                col_dict[col] = f'{feature_group}.{col}'
            feature_df = feature_df.rename(columns=col_dict)

            point_in_time_features = PandasParquetBatchFeaturesRequest.create_point_in_time_feature_df(
                entity_df, feature_df, feature_entities)
            final_df = PandasParquetBatchFeaturesRequest.join_feature_entity_with_main_entity(final_df,
                                                                                              point_in_time_features,
                                                                                              col_dict.values())
        return final_df.reset_index(drop=True)

    @staticmethod
    def read_files_to_pandas(root_location, feature_group, columns):
        path = f'{root_location}/{feature_group}/*/part*'
        file_list = [file.path for file in fsspec.open_files(path)]
        df_list = [pq.read_pandas(f, columns).to_pandas() for f in file_list]
        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def read_feature_group_def(marlin_stub, feature_group):
        feature_group_request_details = FeatureRequestDetails()
        feature_group_request_details.feature_group_name = feature_group
        return marlin_stub.FeatureGroupDefinitionRequest(feature_group_request_details)

    @staticmethod
    def join_feature_entity_with_main_entity(entity_df, point_in_time_feature_df, feature_cols):
        final_output_cols = [e for e in entity_df.columns] + list(feature_cols)
        indexed_point_in_time_feature_df = point_in_time_feature_df.set_index(uuid_str)
        return entity_df.join(indexed_point_in_time_feature_df, how='left', rsuffix='_r')[final_output_cols]

    @staticmethod
    def create_point_in_time_feature_df(entity_df, feature_df, entity_cols):
        entity_df_cols = entity_cols + [target_timestamp_str, uuid_str]
        join_df = entity_df[entity_df_cols].set_index(entity_cols).join(feature_df.set_index(entity_cols), how='inner')

        grouped_df = join_df.groupby(uuid_str)
        agg_df = grouped_df.apply(PandasParquetBatchFeaturesRequest.choose_feature).to_frame(name=event_time_str)
        agg_df[uuid_str] = agg_df.index

        join_df[uuid_str + "_temp"] = join_df[uuid_str]
        final_join_df = join_df.drop(target_timestamp_str, 1).set_index([uuid_str, event_time_str])

        final_df = final_join_df.join(agg_df.set_index([uuid_str, event_time_str]), how='inner', rsuffix='+r')
        return final_df.reset_index(drop=True).rename(columns={uuid_str + "_temp": uuid_str})

    @staticmethod
    def choose_feature(df):
        return df.loc[df[event_time_str] <= df.iloc[0][target_timestamp_str]][event_time_str].max()
