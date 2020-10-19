from functools import lru_cache
from functools import wraps

import grpc
import marlin_service_pb2_grpc
from marlin_service_pb2 import FeatureRequestDetails
from marlin_service_pb2 import FeatureGroupDefinition
from marlin_service_pb2 import FeatureGroupDefinitionRequestDetails
from marlin_service_pb2 import DataType
from marlin_service_pb2 import IngestionMessage
from batch_feature_store.batch_feature_request import PandasParquetBatchFeaturesRequest
import inspect
import configparser
from pathlib import Path


def update_feature_group(func):
    @wraps(func)
    def upate_feature_group_helper(*args, **kwargs):
        if 'feature_group_name' in kwargs:
            kwargs['feature_group_name'] = args[0].client_id + kwargs['feature_group_name']
            return func(*args, **kwargs)
        else:
            index = inspect.getfullargspec(func).args.index('feature_group_name')
            new_args = list(args)
            new_args[index] = new_args[0].client_id + args[index]
            return func(*new_args, **kwargs)

    return upate_feature_group_helper


def dict_set_helper(dict, field):
    for (key, val) in dict.items():
        field[key] = val


def dict_set_helper_with_data_type(dict, field, map_for_data_type):
    for (key, val) in dict.items():
        set_field(map_for_data_type[key], val, field[key])


def set_field(data_type, val, field_to_set):
    if DataType.INTEGER == data_type:
        field_to_set.int_val = val
    elif DataType.LONG == data_type:
        field_to_set.long_val = val
    elif DataType.DOUBLE == data_type:
        field_to_set.double_val = val
    elif DataType.BOOLEAN == data_type:
        field_to_set.bool_val = val
    elif DataType.STRING == data_type:
        field_to_set.string_val = val
    else:
        raise Exception(f'Unkown data type {data_type} for field {field_to_set} and value {val}')


def to_feature_dict(features, schema, feature_group_name):
    feature_dict = {}
    for fk, fv in features.features.items():
        fkn = feature_group_name + '.' + fk
        feature_dict[fkn] = get_feature_value(fv, schema.features[fk])
    return feature_dict


def get_feature_value(fv, data_type):
    if DataType.INTEGER == data_type:
        return fv.int_val
    elif DataType.LONG == data_type:
        return fv.long_val
    elif DataType.DOUBLE == data_type:
        return fv.double_val
    elif DataType.BOOLEAN == data_type:
        return fv.bool_val
    elif DataType.STRING == data_type:
        return fv.string_val
    else:
        raise Exception(f'Unknown data type {data_type} for field {fv}')


class Client(object):
    def __init__(self,
                 server_address,
                 server_port,
                 root_location):
        """Initializer.
                  Creates a gRPC channel for connecting to the server.
                  Adds the channel to the generated client stub.
               Arguments:
                   server_address: host address to marlin server
                   server_port: marlin server port
                   root_location: root directory to marlin store
               Returns:
                   None.
               """
        self.channel = grpc.insecure_channel(f'{server_address}:{server_port}')
        self.stub = marlin_service_pb2_grpc.MarlinServiceStub(self.channel)
        self.batch_store = PandasParquetBatchFeaturesRequest(root_location, self.stub)

    def get_features_as_dict(self, feature_group_name, entities, features):
        return to_feature_dict(self.__get_features_helper(feature_group_name, entities, features).result(),
                               self.__get_feature_group_definition(feature_group_name),
                               feature_group_name.split(self.client_id, 1)[1])

    def get_features(self, feature_group_name, entities, features):
        return self.__get_features_helper(feature_group_name=feature_group_name,
                                          entities=entities, features=features).result()

    def get_features_async(self, feature_group_name, entities, features):
        """Gets a set of features
        Arguments:
            feature_group_name: name of the feature group for which feature data is needed
            entities: dictionary containing values for all the entities in this feature group
            *features: feature list such [f1,f2] which are to be fetched

        Returns:
            returns a key, value - dict {"f1":24,"f2":34}
        """
        return self.__get_features_helper(feature_group_name=feature_group_name, entities=entities, features=features)

    def get_batch_features(self, entity_df, features):
        """ Read batch data as Pandas Dataframe
            Arguments:
                entity_df: Dataframe containing entity and target timestamp
                features: list of features to fetch
        """
        return self.batch_store.get_batch_features(entity_df, features).drop_duplicates()

    def register_feature_group(self,
                               feature_group_name,
                               author,
                               online,
                               offline,
                               source_code,
                               entities,
                               features):
        """ Register a Feature Group
            Arguments:
                feature_group_name: name of the feature group
                author: name of the author for this feature group
                online: flag to indicate whether this features of this feature group need to be available online
                offline: flag to indicate whether this features of this feature group need to be available offline
                source_code: source code or link to source code generating this source code
                entities: dictionary of entities with their data types
                features: dictionary of features with their data types
        """
        feature_group_definitions = FeatureGroupDefinition()
        feature_group_definitions.feature_group_name = feature_group_name
        feature_group_definitions.author = author
        feature_group_definitions.online = online
        feature_group_definitions.offline = offline
        feature_group_definitions.source_code = source_code
        dict_set_helper(entities, feature_group_definitions.entities)
        dict_set_helper(features, feature_group_definitions.features)

        return self.stub.FeatureGroupRegistration(feature_group_definitions)

    def feature_ingest(self, df, entity_name, feature_group_name, event_ts):
        future_list = []
        feature_row = {}
        entity = {}
        for row in df.to_dict(orient='records'):
            for key in row:
                if key in entity_name:
                    entity[key] = row[key]
                else:
                    feature_row[key] = row[key]
            future = self.__ingest_feature_helper(feature_group_name=feature_group_name,
                                                  event_timestamp=event_ts,
                                                  entities=entity,
                                                  features=feature_row)
            future_list.append(future)

        for future in future_list:
            future.result()

    def ingest_features(self,
                        feature_group_name,
                        event_timestamp,
                        entities,
                        features):
        """ Ingest feature to feature store
            Arguments:
                feature_group_name: name of the feature group
                event_timestamp: feature generation timestamp
                entities: dictionary of entities with their data types
                features: dictionary of features with their data types

        """
        return self.__ingest_feature_helper(feature_group_name=feature_group_name, event_timestamp=event_timestamp,
                                            entities=entities, features=features)

    def __get_features_helper(self, feature_group_name, entities, features):
        feature_request = FeatureRequestDetails()
        feature_request.feature_group_name = feature_group_name

        feature_group_definition = self.__get_feature_group_definition(feature_group_name)
        dict_set_helper_with_data_type(entities, feature_request.entities, feature_group_definition.entities)

        feature_request.features_requested.extend(features)

        return self.stub.FeatureRequest.future(feature_request)

    def __ingest_feature_helper(self,
                                feature_group_name,
                                event_timestamp,
                                entities,
                                features):
        ingest_request = IngestionMessage()
        ingest_request.feature_group_name = feature_group_name
        ingest_request.event_timestamp = event_timestamp

        feature_group_definition = self.__get_feature_group_definition(feature_group_name)

        dict_set_helper_with_data_type(entities, ingest_request.entities, feature_group_definition.entities)
        dict_set_helper_with_data_type(features, ingest_request.features, feature_group_definition.features)

        return self.stub.IngestRequest.future(ingest_request)

    @lru_cache(maxsize=None)
    def __get_feature_group_definition(self, feature_group_name):
        return self.stub.FeatureGroupDefinitionRequest(
            FeatureGroupDefinitionRequestDetails(feature_group_name=feature_group_name))


class MarlinServiceClient(Client):
    """
    Client for gRPC functionality
    """

    def __init__(self,
                 server_address,
                 server_port,
                 root_location):
        """Initializer.
           Creates a gRPC channel for connecting to the server.
           Adds the channel to the generated client stub.
        Arguments:
            server_address: host address to marlin server
            server_port: marlin server port
            root_location: root directory to marlin store
        Returns:
            None.
        """
        super().__init__(server_address, server_port, root_location)
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(str(Path.home()) + '/.marlin/config')
        self.client_id = config['default']['client_id']

    @update_feature_group
    def get_features_as_dict(self, feature_group_name, entities, features):
        return super(MarlinServiceClient, self).get_features_as_dict(feature_group_name, entities, features)

    @update_feature_group
    def get_features(self, feature_group_name, entities, features):
        return super(MarlinServiceClient, self).get_features(feature_group_name=feature_group_name,
                                                             entities=entities, features=features)

    @update_feature_group
    def get_features_async(self, feature_group_name, entities, features):
        """Gets a set of features
        Arguments:
            feature_group_name: name of the feature group for which feature data is needed
            entities: dictionary containing values for all the entities in this feature group
            *features: feature list such [f1,f2] which are to be fetched

        Returns:
            returns a key, value - dict {"f1":24,"f2":34}
        """
        return super(MarlinServiceClient, self).get_features_async(feature_group_name=feature_group_name,
                                                                   entities=entities, features=features)

    @update_feature_group
    def register_feature_group(self,
                               feature_group_name,
                               author,
                               online,
                               offline,
                               source_code,
                               entities,
                               features):
        """ Register a Feature Group
            Arguments:
                feature_group_name: name of the feature group
                author: name of the author for this feature group
                online: flag to indicate whether this features of this feature group need to be available online
                offline: flag to indicate whether this features of this feature group need to be available offline
                source_code: source code or link to source code generating this source code
                entities: dictionary of entities with their data types
                features: dictionary of features with their data types
        """

        return super(MarlinServiceClient, self).register_feature_group(feature_group_name, author, online, offline,
                                                                       source_code, entities, features)

    @update_feature_group
    def feature_ingest(self, df, entity_name, feature_group_name, event_ts):
        return super(MarlinServiceClient, self).feature_ingest(df, entity_name, feature_group_name, event_ts)

    @update_feature_group
    def ingest_features(self,
                        feature_group_name,
                        event_timestamp,
                        entities,
                        features):
        """ Ingest feature to feature store
            Arguments:
                feature_group_name: name of the feature group
                event_timestamp: feature generation timestamp
                entities: dictionary of entities with their data types
                features: dictionary of features with their data types

        """
        return super(MarlinServiceClient, self).ingest_features(feature_group_name=feature_group_name,
                                                                event_timestamp=event_timestamp,
                                                                entities=entities, features=features)

    def get_batch_features(self, entity_df, features):
        """ Read batch data as Pandas Dataframe
            Arguments:
                entity_df: Dataframe containing entity and target timestamp
                features: list of features to fetch
        """
        modified_features = []
        dict = {}
        for feature_def in features:
            split = feature_def.split(':', 1)
            modified_features.append(self.client_id + split[0] + ":" + split[1])
            dict[self.client_id + split[0] + "." + split[1]] = split[0] + "." + split[1]
        df = super(MarlinServiceClient, self).get_batch_features(entity_df, modified_features)

        return df.rename(columns=dict)
