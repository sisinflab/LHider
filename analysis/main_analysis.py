from analysis.score_analysis import ScoreAnalyzer

datasets = ['facebook_books', 'yahoo_movies']
data_types = ['clean']
# score_types = ['jaccard', 'euclidean', 'manhattan']
score_types = ['jaccard', 'euclidean', 'manhattan']
epss = ['0.5', '1.0', '2.0', '3.0', '5.0', '10.0', "random"]
# epss = ['random']
decimals = [4]
metrics = ['min', 'max', 'mean', 'std']

for d in datasets:
    for d_t in data_types:
        for s_t in score_types:
            for e in epss:
                for dec in decimals:
                    analyzer = ScoreAnalyzer(dataset_name=d, data_type=d_t, score_type=s_t,
                                             eps=e)
                    # analyzer.compare_metric_with_manhattan(store_plot=True, store_stats=True)
                    # analyzer.over_generation_utility(store_plot=True, store_stats=True)
                    analyzer.score_distribution(decimal=dec, store_plot=True, store_stats=True)
                    # analyzer.metrics_over_generation(store_stats=True)

                    # for m in metrics:
                    #     analyzer.plot_metrics_over_generation(metrics=[m], store_plot=True)
