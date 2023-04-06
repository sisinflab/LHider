import pickle
import os


def multi_aggregation_on_results():
    folder = '/home/alberto/PycharmProjects/ExponentialMechanismForRecommenderSystems/results/aggregate_results'
    results = list(os.listdir(folder))
    file_extension = '.pk'
    results = [os.path.join(folder, r) for r in results if file_extension in r]
    for r in results:
        print(f'file found: {r}')

    print('aggregate results')
    aggregation = dict()
    for path in results:
        with open(path, 'rb') as file:
            result = pickle.load(file)
            print(f'Found {len(result)} scores')
            aggregation.update(result)

    print('storing aggregating results')
    final_results_folder = '/home/alberto/PycharmProjects/ExponentialMechanismForRecommenderSystems/results/final'
    if not os.path.exists(final_results_folder):
        os.makedirs(final_results_folder)

    aggregate_result_path = os.path\
        .join(final_results_folder, f'{min(aggregation.keys())}_{max(aggregation.keys())}_n{len(aggregation)}.pk')
    with open(aggregate_result_path, 'wb') as result_file:
        pickle.dump(aggregation, result_file)
    print(f'results stored at {aggregate_result_path}')
