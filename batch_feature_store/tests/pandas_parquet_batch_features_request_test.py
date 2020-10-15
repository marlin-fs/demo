import pandas as pd
from batch_feature_store.batch_feature_request import PandasParquetBatchFeaturesRequest
from pandas.testing import assert_frame_equal
import math
from mock import patch
import uuid

def test_create_point_in_time_feature_df():
    uuid1  = uuid.uuid4()
    uuid2 = uuid.uuid4()
    uuid3 = uuid.uuid4()
    uuid4 = uuid.uuid4()
    uuid5 = uuid.uuid4()
    entity_df = pd.DataFrame([
        [1, 1, 1, 1, 1, uuid1],
        [1, 1, 1, 1, 3, uuid2],
        [1, 1, 1, 1, 5, uuid3],
        [1, 1, 1, 1, 9, uuid4],
        [1, 3, 1, 1, 3, uuid5]
    ], columns=['A', 'B', 'C', 'D', 'target_timestamp', 'uuid'])

    feature_df = pd.DataFrame([
        [1, 1, 3, 3, 3],
        [1, 1, 4, 4, 4],
        [1, 1, 8, 8, 8],
        [2, 1, 8, 8, 8]
    ], columns=['A', 'B', 'event_time', 'feature_1', 'feature_2'])
    output = PandasParquetBatchFeaturesRequest.create_point_in_time_feature_df(entity_df, feature_df, ['A', 'B'])\
        .reset_index(drop=True).sort_values(by=['uuid'])
    expected_df = pd.DataFrame([
        [uuid2, 3, 3],
        [uuid3, 4, 4],
        [uuid4, 8, 8]
    ], columns=['uuid', 'feature_1', 'feature_2'])

    sorted_expected = expected_df.sort_values(by=['uuid']).reset_index(drop=True)
    sorted_output = output.sort_values(by=['uuid']).reset_index(drop=True)

    assert_frame_equal(sorted_expected.reindex(sorted(expected_df.columns), axis=1),
                       sorted_output.reindex(sorted(expected_df.columns), axis=1))


def test_join_feature_entity_df_with_main_entity_df():
    entity_df = pd.DataFrame([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 3, 2],
        [1, 1, 1, 1, 5, 3],
        [1, 1, 1, 1, 9, 4],
        [1, 3, 1, 1, 3, 5]
    ], columns=['A', 'B', 'C', 'D', 'target_timestamp', 'uuid'])

    feature_df = pd.DataFrame([
        [1, 1, 3, 2, 3, 3],
        [1, 1, 5, 3, 4, 4],
        [1, 1, 9, 4, 8, 8]
    ], columns=['A', 'B', 'target_timestamp', 'uuid', 'feature_1', 'feature_2'])

    output = PandasParquetBatchFeaturesRequest.join_feature_entity_with_main_entity(entity_df.set_index('uuid'),
                                                                                    feature_df,
                                                                                    ['feature_1', 'feature_2'])

    expected_df = pd.DataFrame([
        [1, 1, 1, 1, 1, 1, math.nan, math.nan],
        [1, 1, 1, 1, 3, 2, 3, 3],
        [1, 1, 1, 1, 5, 3, 4, 4],
        [1, 1, 1, 1, 9, 4, 8, 8],
        [1, 3, 1, 1, 3, 5, math.nan, math.nan]
    ], columns=['A', 'B', 'C', 'D', 'target_timestamp', 'uuid', 'feature_1', 'feature_2'])

    assert_frame_equal(expected_df.reindex(sorted(expected_df.columns), axis=1),
                       output.reset_index().reindex(sorted(output.reset_index()), axis=1))


@patch('batch_feature_store.batch_feature_request.PandasParquetBatchFeaturesRequest.read_files_to_pandas')
@patch('batch_feature_store.batch_feature_request.PandasParquetBatchFeaturesRequest.read_feature_group_def')
def test_get_batch_features_helper(read_feature_group_def, read_files_to_pandas):
    read_files_to_pandas.side_effect = mocked_read_files_to_pandas
    read_feature_group_def.side_effect = mocked_get_feature_group
    # entity_df, feature_list, root_location, marlin_stub
    # root_location, feature_group, columns

    entity_df = pd.DataFrame([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 3],
        [1, 1, 1, 1, 5],
        [1, 1, 1, 1, 9],
        [1, 3, 1, 3, 3]
    ], columns=['A', 'B', 'C', 'D', 'target_timestamp'])
    output = PandasParquetBatchFeaturesRequest("", None).get_batch_features(entity_df, ["f:feature_1", "f:feature_2",
                                                                                        "f1:feature_1", "f1:feature_4"])
    expected_df = pd.DataFrame([
        [1, 1, 1, 1, 1, math.nan, math.nan, math.nan, math.nan],
        [1, 1, 1, 1, 3, 3, 3, 6, 3],
        [1, 1, 1, 1, 5, 4, 4, 8, 4],
        [1, 1, 1, 1, 9, 8, 8, 16, 8],
        [1, 3, 1, 3, 3, math.nan, math.nan, math.nan, math.nan]
    ], columns=['A', 'B', 'C', 'D', 'target_timestamp', 'f.feature_1', 'f.feature_2', 'f1.feature_1', 'f1.feature_4'])
    assert_frame_equal(expected_df, output)


def mocked_read_files_to_pandas(root_location, feature_group, columns):
    if feature_group == "f":
        return pd.DataFrame([
            [1, 1, 2, 3, 3, 3],
            [1, 1, 2, 4, 4, 4],
            [1, 1, 2, 8, 8, 8]
        ], columns=['A', 'B', 'create_timestamp', 'event_time', 'feature_1', 'feature_2'])
    else:
        return pd.DataFrame([
            [1, 1, 2, 3, 6, 3],
            [1, 1, 2, 4, 8, 4],
            [1, 1, 2, 8, 16, 8]
        ], columns=['C', 'D', 'create_timestamp', 'event_time', 'feature_1', 'feature_4'])


def mocked_get_feature_group(marlin_stub, feature_group):
    if feature_group == "f":
        return EntityShell({"A": 1, "B": 1})
    else:
        return EntityShell({"C": 1, "D": 1})


class EntityShell:
    def __init__(self, entities):
        self.entities = entities
