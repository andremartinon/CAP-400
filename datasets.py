import gc
import numpy as np

from pathlib import Path
from timeit import default_timer as timer

from cml import CML, ToroidalBoundaryCondition, SteadyBoundaryCondition,\
    RandomInitialCondition, GaussianInitialCondition, LogisticMap, NoiseMap, \
    FourNeighborCoupling, Evolution, StatisticalMomentsMetric, GradientMetric, \
    EntropyMetric, EulerCharacteristicMetric, GPAMetric, create_output_dir
from lstm import TimeSeriesLSTM


def train_time_series(time_series: np.ndarray, time_series_name: str,
                      dataset_name: str, output_dir: Path):
    title = f'{time_series_name} Training [{dataset_name}]'
    print(title)
    lstm = TimeSeriesLSTM(time_series)
    lstm.train(epochs=50, batch_size=1, verbose=0)
    file_name = (f'{dataset_name}_lstm_'
                 f'{time_series_name.lower().replace(" ", "_")}_train.png')
    lstm.plot(show=False, title=title,
              y_label=f'{time_series_name}',
              file_name=output_dir / file_name)


if __name__ == '__main__':
    start_total = timer()

    output_root_dir = Path('/tmp/cml/cml_steady/')

    ic = RandomInitialCondition(seed=2021)
    bc = SteadyBoundaryCondition()

    mapping = LogisticMap(3.9)
    datasetcml1 = CML(ic, bc, mapping, FourNeighborCoupling(0.1))
    datasetcml2 = CML(ic, bc, mapping, FourNeighborCoupling(0.5))
    datasetcml3 = CML(ic, bc, mapping, FourNeighborCoupling(0.9))

    mapping = NoiseMap(alpha=1.0, noise_contribution=0.1)
    datasetnoise1 = CML(ic, bc, mapping, FourNeighborCoupling(0.1))
    datasetnoise2 = CML(ic, bc, mapping, FourNeighborCoupling(0.5))
    datasetnoise3 = CML(ic, bc, mapping, FourNeighborCoupling(0.9))

    ic = GaussianInitialCondition()
    datasetcml4 = CML(ic, bc, LogisticMap(3.9), FourNeighborCoupling(0.5),
                      grid_size=128)

    datasetnoise4 = CML(ic, bc, NoiseMap(alpha=0.0, noise_contribution=0.1),
                        FourNeighborCoupling(0.9), grid_size=128)

    cml_datasets = [(datasetcml1, 1024, 'datasetcml1', 1),
                    (datasetcml2, 1024, 'datasetcml2', 1),
                    (datasetcml3, 1024, 'datasetcml3', 1),
                    (datasetcml4, 2048, 'datasetcml4', 4),
                    (datasetnoise1, 1024, 'datasetnoise1', 1),
                    (datasetnoise2, 1024, 'datasetnoise2', 1),
                    (datasetnoise3, 1024, 'datasetnoise3', 1),
                    (datasetnoise4, 2048, 'datasetnoise4', 4)]

    for cml, iterations, name, gradient_steps in cml_datasets:
        start = timer()
        output_dir = output_root_dir / name
        create_output_dir(output_dir)

        time_evolution = Evolution(cml,
                                   iterations=iterations,
                                   output_dir=output_dir,
                                   dataset_name=name)

        time_evolution.run()

        time_evolution.save_csv()
        time_evolution.plot(show=False)
        time_evolution.plot(show=False, binary_threshold=0.68)
        time_evolution.animate(show=False)

        print('Start stat_metrics')
        stat_metrics = StatisticalMomentsMetric(time_evolution)
        stat_metrics.measure()
        stat_metrics.plot(show=False)
        stat_metrics.save()
        for ts, ts_name in [(stat_metrics.metrics[:, 0], 'Skewness'),
                            (stat_metrics.metrics[:, 1], 'Kurtosis'),
                            (stat_metrics.metrics[:, 2], 'Variance')]:
            train_time_series(ts, ts_name, name, output_dir)
        del stat_metrics
        gc.collect()

        print('Start grad_metrics')
        grad_metrics = GradientMetric(time_evolution)
        grad_metrics.measure()
        grad_metrics.plot(10, 2, show=False)
        grad_metrics.plot_four_gradients(step=gradient_steps, show=False)
        grad_metrics.animate(step=gradient_steps, show=False)
        grad_metrics.save()
        del grad_metrics
        gc.collect()

        print('Start entropy_metrics')
        entropy_metrics = EntropyMetric(time_evolution)
        entropy_metrics.measure(bins=256)
        entropy_metrics.plot(show=False)
        entropy_metrics.save()
        for ts, ts_name in [(entropy_metrics.metrics[:, 1], 'Modulus'),
                            (entropy_metrics.metrics[:, 2], 'Phases')]:
            train_time_series(ts, ts_name, name, output_dir)
        del entropy_metrics
        gc.collect()

        print('Start euler_metrics')
        euler_metrics = EulerCharacteristicMetric(time_evolution)
        euler_metrics.measure(threshold=0.68, connectivity=1)
        euler_metrics.plot(show=False)
        euler_metrics.save()
        for ts, ts_name in \
                [(euler_metrics.metrics[:, 1], 'Euler Characteristic')]:
            train_time_series(ts, ts_name, name, output_dir)
        del euler_metrics
        gc.collect()

        print('Start gpa_metrics')
        gpa_metrics = GPAMetric(time_evolution)
        gpa_metrics.measure(tolerance=0.01)
        gpa_metrics.plot(show=False)
        gpa_metrics.save()
        for ts, ts_name in \
                [(gpa_metrics.metrics[:, 1], 'GPA G2')]:
            train_time_series(ts, ts_name, name, output_dir)
        print(f'{name} elapsed time (s)', timer() - start)

    print(f'Total elapsed time (s)', timer() - start_total)
