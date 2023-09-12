import ipaddress
import pandas as pd
from decimal import Decimal


def get_dataset(path: str) -> pd.DataFrame:
    """
    Reads CSV logfile to pandas dataframe
    :rtype: pd.DataFrame
    :param path: Path to Logfile
    :return: Pandas DataFrame with logs
    """
    colnames = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
                "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
                "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
                "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
                "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm",
                "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"]

    dataset_raw: pd.DataFrame = pd.read_csv(path, names=colnames, header=None, sep=",", error_bad_lines=False,
                                            warn_bad_lines=True, dtype={"sport": str, "dsport": str, "attack_cat": str})
    return dataset_raw


def dataframe_to_numeric(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Converts non-numeric types to numeric types using embedding
    :rtype: pd.DataFrame
    :param dataset: Logfile
    :return: Logfile with numeric embeddings
    """

    state_dict = {k: v for v, k in enumerate(dataset['state'].unique())}
    proto_dict = {k: v for v, k in enumerate(dataset['proto'].unique())}
    service_dict = {k: v for v, k in enumerate(dataset['service'].unique())}

    dataset.srcip = dataset.srcip.apply(lambda x: int(ipaddress.IPv4Address(x)))
    dataset.dstip = dataset.dstip.apply(lambda x: int(ipaddress.IPv4Address(x)))
    dataset.sport = dataset.sport.apply(lambda x: hex_int_conversion(x))
    dataset.dsport = dataset.dsport.apply(lambda x: hex_int_conversion(x))
    dataset.proto.replace(to_replace=proto_dict.keys(), value=proto_dict.values(), inplace=True)
    dataset.state.replace(to_replace=state_dict.keys(), value=state_dict.values(), inplace=True)
    dataset.service.replace(to_replace=service_dict.keys(), value=service_dict.values(), inplace=True)

    return dataset


def create_train_test_datasets(dataset: pd.DataFrame, supervised: bool = False, ratio: Decimal = 1,
                               globalset: bool = False) -> (pd.DataFrame, pd.DataFrame):
    """
    Creates Test and Training Datasets
    :rtype: (pd.DataFrame, pd.Dataframe)
    :param dataset: Logfile
    :param supervised: if false remove anomalies from traindata
    :param ratio: amount of avalable traindata
    :param globalset: true, merge all dicts for dataset including all devices
    :return: Test and Trainset
    """
    unique_int_ips = pd.concat([dataset['srcip'], dataset['dstip']], ignore_index=True).unique()

    # Dataframe dict containing the dicts for every ip
    data_frame_dict = {ip: pd.DataFrame() for ip in unique_int_ips}
    train_frame_dict = {ip: pd.DataFrame() for ip in unique_int_ips}
    test_frame_dict = {ip: pd.DataFrame() for ip in unique_int_ips}

    for key in data_frame_dict.keys():
        data_frame_dict[key] = dataset[:][
            ((dataset.srcip == key) | (dataset.dstip == key))]

    for key in data_frame_dict.keys():
        split = int(len(data_frame_dict[key]) * 0.8)
        if not supervised:
            # Only select packets with no anomaly
            train_frame_dict[key] = data_frame_dict[key][:int(split * ratio)][data_frame_dict[key].Label == 0]
        if supervised:
            train_frame_dict[key] = data_frame_dict[key][:int(split * ratio)]
        test_frame_dict[key] = data_frame_dict[key][split:]

    if globalset:
        # Dataset merge for global model

        global_train_set: pd.DataFrame = pd.concat(
            [train_frame_dict[i] for i in list(train_frame_dict.keys())]).sort_index()

        global_train_set = global_train_set[~global_train_set.index.duplicated(keep='first')]

        global_test_set = pd.concat([test_frame_dict[i] for i in list(test_frame_dict.keys())]).sort_index()

        return global_train_set, global_test_set

    return train_frame_dict, test_frame_dict


# ---------------Helpers-----------#
def hex_int_conversion(value):
    try:
        return int(value)
    except ValueError:
        if value == '-':
            return 0
        return int(value, 0)
