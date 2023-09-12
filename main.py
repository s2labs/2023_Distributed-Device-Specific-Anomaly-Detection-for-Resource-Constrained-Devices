import decimal
import ipaddress
import multiprocessing as mp
import os
import shutil
import time
from datetime import datetime

# Local Import
import preprocess
import training

# Three options: "distributed", "global", "pi"
if __name__ == '__main__':
    option = "distributed"
    # option = "global"
    # option = "pi"

    print(f'Training of {option} models')
    dataset_raw = preprocess.get_dataset("UNSW-NB15_1.csv")
    dataset_raw = preprocess.dataframe_to_numeric(dataset_raw)

    print("Number of processors: ", mp.cpu_count())

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d:%H-%M-%S")

    os.mkdir(str(f'{date_time}'))

    sourcefolders = ["models"]

    # Initialize dataset ratios
    z = [decimal.Decimal(i) / decimal.Decimal(10) for i in range(10, 0, -1)]

    ctx = mp.get_context('spawn')

    # Start training loops
    __start_time = time.time()

    for ratio in z:
        print(ratio)
        os.mkdir(str(f'{date_time}/{ratio}'))
        os.mkdir(str(f'{date_time}/{ratio}/models'))
        os.mkdir(str(f'{date_time}/{ratio}/prints'))
        results = []

        # For Pi train largest device
        if option == 'pi':
            _TrainFrameDict, _TestFrameDict = preprocess.create_train_test_datasets(dataset_raw, True, ratio)
            pidevice = int(ipaddress.IPv4Address("59.166.0.2"))
            _TrainFrameDict[pidevice].to_csv(f'{date_time}/{ratio}/pidata.csv', index=False)

        # For distributed train all models, use all CPU cores for parallel processing
        if option == 'distributed':
            _TrainFrameDict, _TestFrameDict = preprocess.create_train_test_datasets(dataset_raw, True, ratio)

            with ctx.Pool(mp.cpu_count()) as pool:
                results = pool.starmap(training.do_training,
                                       zip(list(_TrainFrameDict.keys()), _TrainFrameDict.values(),
                                           _TestFrameDict.values()))

        # For global train one model on complete dataset
        if option == 'global':
            _TrainFrameDict, _TestFrameDict = preprocess.create_train_test_datasets(dataset_raw, True, ratio, True)

            results = training.do_training(0, _TrainFrameDict, _TestFrameDict)

        # Store results
        destination = str(f'{date_time}/{ratio}/')

        for folder in sourcefolders:

            allfiles = os.listdir(folder)

            for f in allfiles:
                src_path = os.path.join(f'{folder}', f)
                dst_path = os.path.join(f'{destination}{folder}', f)
                shutil.move(src_path, dst_path)
        print(f'Successful runs (0:success 1:error): {results}')
    if option == 'distributed':
        pool.close()
    __end_time = time.time()

    with open(f'{date_time}-total_time.txt', 'a') as the_file:
        the_file.write(str(__end_time - __start_time))
