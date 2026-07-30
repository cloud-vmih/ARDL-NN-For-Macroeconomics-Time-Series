from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import (
    ARDLSelectionConfig,
    DataConfig,
    ExperimentConfig,
    FFNNConfig,
    StationarityConfig,
    VMDConfig,
)
from .experiment import VMDARDLFFNNExperiment
from .models.varnn import VARNNConfig, run_varnn_pipeline


def main() -> None:
    """Phân tích tham số CLI và chạy pipeline được chọn."""
    parser = argparse.ArgumentParser(description="Run VMD + ARDL order selection + FFNN grid search.")
    parser.add_argument("--data", required=True, help="CSV input path.")
    parser.add_argument("--out", default="results", help="Output directory.")
    parser.add_argument("--date-col", default="TIME_PERIOD")
    parser.add_argument("--target", default="Export_US")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--freq", default="MS")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--vmd-modes", type=int, default=3)
    parser.add_argument(
        "--vmd-backend",
        choices=["self-contained", "library"],
        default="self-contained",
        help="Use the package's self-contained VMD or an optional third-party VMD library.",
    )
    parser.add_argument("--max-target-lag", type=int, default=12)
    parser.add_argument("--max-exog-lag", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--target-lag-strategy",
        choices=["validation-screen", "joint-validation-screen", "ic-topk", "fixed", "ardl-select-order", "ardl-select-order-joint"],
        default="ardl-select-order",
        help="How target autoregressive lags are selected before FFNN search.",
    )
    parser.add_argument("--fixed-target-lags", type=int, nargs="+", default=[1, 12])
    parser.add_argument("--target-lag-top-n", type=int, default=2)
    parser.add_argument(
        "--target-lag-preset",
        choices=["acf-pacf", "monthly", "daily", "yearly"],
        default="acf-pacf",
        help="Preset target-lag candidate sets for validation-screen.",
    )
    parser.add_argument("--target-lag-acf-pacf-top-n", type=int, default=3)
    parser.add_argument("--target-lag-acf-pacf-max-lags-per-set", type=int, default=3)
    parser.add_argument("--no-force-target-lag-1", action="store_true")
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Hidden layer counts to search when using n_features-based FFNN widths.",
    )
    parser.add_argument(
        "--hidden-width-multipliers",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25],
        help="Width multipliers applied to n_features for each hidden layer.",
    )
    parser.add_argument(
        "--hr",
        type=int,
        nargs="+",
        default=None,
        help="Legacy absolute one-hidden-layer neuron grid. Overrides n_features-based widths when set.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh"],
        default="tanh",
        help="Activation used for fast screening and as the fallback FFNN activation.",
    )
    parser.add_argument(
        "--activation-grid",
        choices=["relu", "tanh"],
        nargs="+",
        default=None,
        help="Activation functions to tune in the FFNN hyperparameter search.",
    )
    parser.add_argument(
        "--no-activation-tuning",
        action="store_true",
        help="Disable activation tuning and use only --activation in the FFNN search.",
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--ffnn-learning-rate-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.001],
        help="Learning rates compared on validation during the FFNN hyperparameter search.",
    )
    parser.add_argument(
        "--search-strategy",
        choices=["staged-halving", "full-grid"],
        default="staged-halving",
        help="FFNN search budget strategy.",
    )
    parser.add_argument(
        "--lag-spec-strategy",
        choices=["staged", "full-product"],
        default="staged",
        help="How ARDL top lags are converted into FFNN lag specs.",
    )
    parser.add_argument("--max-lag-specs", type=int, default=16)
    parser.add_argument("--top-k-lag-specs", type=int, default=4)
    parser.add_argument("--top-component-candidates", type=int, default=3)
    parser.add_argument(
        "--lag-screen-folds",
        type=int,
        default=3,
        help="Number of train-only expanding folds used to fast-screen lag specs.",
    )
    parser.add_argument(
        "--lag-screen-validation-size",
        type=int,
        default=12,
        help="Number of timestamps in each train-only lag-screen validation block.",
    )
    parser.add_argument(
        "--lag-screen-learning-rate-grid",
        type=float,
        nargs="+",
        default=None,
        help="Fast-screen learning rates; defaults to --ffnn-learning-rate-grid.",
    )
    parser.add_argument(
        "--lag-screen-selection-rule",
        choices=["top-k", "one-standard-error"],
        default="one-standard-error",
        help="How rolling lag-screen results enter the expensive FFNN grid.",
    )
    parser.add_argument(
        "--final-selection-scale",
        choices=["level", "transformed"],
        default="level",
        help="Scale used to select the final reconstructed validation winner.",
    )
    parser.add_argument("--fast-max-iter", type=int, default=150)
    parser.add_argument("--fast-hr", type=int, default=None)
    parser.add_argument("--fast-hidden-layers", type=int, default=1)
    parser.add_argument("--fast-hidden-width-multiplier", type=float, default=1.0)
    parser.add_argument("--fast-alpha", type=float, default=1e-3)
    parser.add_argument("--no-stationarity", action="store_true")
    parser.add_argument(
        "--pipeline",
        choices=[
            "vmd",
            "no-vmd",
            "both",
            "varnn",
            "model-comparison",
            "benchmark",
            "block-proposed-benchmark",
            "sensitivity",
            "vmd-ardl-select-order",
            "vmd-ardl-joint-select",
            "vmd-ardl-hard-ffnn",
        ],
        default="vmd",
        help=(
            "Forecasting pipeline to run. 'both' compares no-VMD vs VMD; "
            "'model-comparison' compares VMD-ARDL-FFNN vs VARNN; "
            "'benchmark' runs the complete paper benchmark, A0-A5 ablation and DM tests; "
            "'block-proposed-benchmark' promotes the AIC block-subset two-stage "
            "workflow to Proposed, uses shared mean-seed model selection, and runs A0-A7; "
            "'vmd-ardl-hard-ffnn' full-grids top-3 ARDL lag orders while assigning "
            "one train-only frequency/lag-based FFNN configuration per candidate; "
            "'sensitivity' runs the predeclared one-factor-at-a-time robustness grid; "
            "'vmd-ardl-select-order' runs VMD-ARDL-FFNN using statsmodels ardl_select_order for target lag selection; "
            "'vmd-ardl-joint-select' runs VMD-ARDL-FFNN using statsmodels ardl_select_order for both target and exogenous lag selection."
        ),
    )
    parser.add_argument("--varnn-lag-criterion", choices=["aic", "bic", "hqic", "fpe"], default="aic")
    parser.add_argument("--varnn-maxlags", type=int, default=None)
    parser.add_argument("--varnn-min-lag", type=int, default=1)
    parser.add_argument(
        "--varnn-fixed-lag",
        type=int,
        default=0,
        help="Fixed VAR lag. Use 0 to select by --varnn-lag-criterion.",
    )
    parser.add_argument("--varnn-max-lag", type=int, default=2, help="Maximum allowed dynamically selected VAR lag.")
    parser.add_argument("--varnn-hidden-units", type=int, default=32)
    parser.add_argument("--varnn-epochs", type=int, default=500)
    parser.add_argument("--varnn-batch-size", type=int, default=32)
    parser.add_argument("--varnn-patience", type=int, default=12)
    parser.add_argument("--varnn-seed", type=int, default=7)
    parser.add_argument("--paper-seeds", type=int, nargs="+", default=[7, 42, 123, 2024, 2025])
    parser.add_argument("--skip-varnn", action="store_true")
    parser.add_argument(
        "--block-subset-search-strategy",
        choices=["full-grid", "two-stage"],
        default="full-grid",
        help=(
            "Search strategy for the legacy benchmark's AIC block-subset comparator. "
            "block-proposed-benchmark always runs both A5 full-grid and A7 two-stage. "
            "two-stage keeps four lag specs/component after a lightweight "
            "alpha=0.1, learning-rate=0.001 screen."
        ),
    )
    parser.add_argument("--varnn-tune", action="store_true", help="Enable VARNN hyperparameter tuning.")
    parser.add_argument("--varnn-hidden-units-grid", type=int, nargs="+", default=[8, 16, 32, 64], help="Hidden units candidates for VARNN tuning.")
    parser.add_argument("--varnn-activation", choices=["sigmoid", "tanh", "relu"], default="relu", help="Default activation for VARNN.")
    parser.add_argument("--varnn-activation-grid", choices=["sigmoid", "tanh", "relu"], nargs="+", default=["sigmoid", "tanh", "relu"], help="Activation candidates for VARNN tuning.")
    parser.add_argument("--varnn-learning-rate", type=float, default=1e-3, help="Default learning rate for VARNN.")
    parser.add_argument("--varnn-learning-rate-grid", type=float, nargs="+", default=[1e-3, 1e-2], help="Learning rate candidates for VARNN tuning.")
    args = parser.parse_args()
    default_ffnn = FFNNConfig()
    activation = args.activation or default_ffnn.activation
    if args.no_activation_tuning and args.activation_grid is not None:
        parser.error("--no-activation-tuning cannot be used together with --activation-grid.")
    if args.no_activation_tuning:
        activation_grid = (activation,)
    else:
        activation_grid = tuple(args.activation_grid) if args.activation_grid is not None else default_ffnn.activation_grid

    config = ExperimentConfig(
        data=DataConfig(
            date_col=args.date_col,
            target=args.target,
            features=tuple(args.features),
            freq=args.freq if args.freq.lower() != "none" else None,
            log_transform=not args.no_log,
        ),
        vmd=replace(VMDConfig(), modes=args.vmd_modes, backend=args.vmd_backend.replace("-", "_")),
        ardl=replace(
            ARDLSelectionConfig(),
            max_target_lag=args.max_target_lag,
            max_exog_lag=args.max_exog_lag,
            top_n=args.top_n,
            target_lag_strategy=args.target_lag_strategy.replace("-", "_"),
            fixed_target_lags=tuple(args.fixed_target_lags),
            target_lag_top_n=args.target_lag_top_n,
            target_lag_preset=args.target_lag_preset.replace("-", "_"),
            target_lag_acf_pacf_top_n=args.target_lag_acf_pacf_top_n,
            target_lag_acf_pacf_max_lags_per_set=args.target_lag_acf_pacf_max_lags_per_set,
            force_target_lag_1=not args.no_force_target_lag_1,
            lag_spec_strategy=args.lag_spec_strategy.replace("-", "_"),
            max_lag_specs=args.max_lag_specs,
        ),
        ffnn=replace(
            default_ffnn,
            hidden_layer_candidates=tuple(args.hidden_layers),
            hidden_width_multipliers=tuple(args.hidden_width_multipliers),
            hidden_units_candidates=tuple(args.hr) if args.hr is not None else tuple(),
            activation=activation,
            activation_grid=activation_grid,
            learning_rate_grid=tuple(args.ffnn_learning_rate_grid),
            max_iter=args.max_iter,
            search_strategy=args.search_strategy.replace("-", "_"),
            fast_hidden_units=args.fast_hr,
            fast_hidden_layers=args.fast_hidden_layers,
            fast_hidden_width_multiplier=args.fast_hidden_width_multiplier,
            fast_alpha=args.fast_alpha,
            fast_max_iter=args.fast_max_iter,
            top_k_lag_specs=args.top_k_lag_specs,
            top_component_candidates=args.top_component_candidates,
            lag_screen_folds=args.lag_screen_folds,
            lag_screen_validation_size=args.lag_screen_validation_size,
            lag_screen_learning_rate_grid=(
                tuple(args.lag_screen_learning_rate_grid)
                if args.lag_screen_learning_rate_grid is not None
                else tuple()
            ),
            lag_screen_selection_rule=args.lag_screen_selection_rule.replace("-", "_"),
            final_selection_scale=args.final_selection_scale,
        ),
        stationarity=replace(StationarityConfig(), enabled=not args.no_stationarity),
        output_dir=Path(args.out),
    )
    experiment = VMDARDLFFNNExperiment(config)

    varnn_config = VARNNConfig(
        lag_criterion=args.varnn_lag_criterion,
        maxlags=args.varnn_maxlags,
        min_lag=args.varnn_min_lag,
        max_lag=args.varnn_max_lag,
        fixed_lag=None if args.varnn_fixed_lag == 0 else args.varnn_fixed_lag,
        hidden_units=args.varnn_hidden_units,
        epochs=args.varnn_epochs,
        batch_size=args.varnn_batch_size,
        patience=args.varnn_patience,
        seed=args.varnn_seed,
        tune=args.varnn_tune,
        hidden_units_grid=tuple(args.varnn_hidden_units_grid),
        activation=args.varnn_activation,
        activation_grid=tuple(args.varnn_activation_grid),
        learning_rate=args.varnn_learning_rate,
        learning_rate_grid=tuple(args.varnn_learning_rate_grid),
    )

    if args.pipeline == "varnn":
        try:
            result = run_varnn_pipeline(
                args.data,
                config,
                varnn_config,
            )
        except ImportError as exc:
            parser.error(str(exc))
        print("VARNN metrics:")
        print(result["metrics"].to_string(index=False))
    elif args.pipeline == "no-vmd":
        result = experiment.run_without_vmd(args.data)
        print("Best model rows:")
        print(result["best_component_models"].to_string(index=False))
        print("\nFinal metrics:")
        print(result["final_metrics"].to_string(index=False))
    elif args.pipeline == "both":
        result = experiment.run_comparison(args.data)
        print("VMD vs no-VMD comparison metrics:")
        print(result["comparison_metrics"].to_string(index=False))
    elif args.pipeline == "model-comparison":
        try:
            result = experiment.run_model_comparison(
                args.data,
                varnn_config,
            )
        except ImportError as exc:
            parser.error(str(exc))
        print("VMD-ARDL-FFNN vs VARNN comparison metrics:")
        print(result["comparison_metrics"].to_string(index=False))
    elif args.pipeline == "benchmark":
        result = experiment.run_benchmark(
            args.data,
            varnn_config,
            seeds=tuple(args.paper_seeds),
            include_varnn=not args.skip_varnn,
            block_subset_search_strategy=args.block_subset_search_strategy.replace(
                "-", "_"
            ),
        )
        print("Benchmark primary per-seed mean/std metrics:")
        print(result["comparison_metrics"].to_string(index=False))
        print("\nBenchmark secondary ensemble mean-prediction metrics:")
        print(result["ensemble_comparison_metrics"].to_string(index=False))
    elif args.pipeline == "block-proposed-benchmark":
        result = experiment.run_block_proposed_benchmark(
            args.data,
            varnn_config,
            seeds=tuple(args.paper_seeds),
            include_varnn=not args.skip_varnn,
        )
        print("Block-proposed benchmark primary per-seed mean/std metrics:")
        print(result["comparison_metrics"].to_string(index=False))
        print("\nBlock-proposed benchmark secondary ensemble mean-prediction metrics:")
        print(result["ensemble_comparison_metrics"].to_string(index=False))
    elif args.pipeline == "vmd-ardl-hard-ffnn":
        hard_config = replace(
            config,
            ardl=replace(
                config.ardl,
                ic="aic",
                target_lag_strategy="ardl_select_order",
                exog_lag_strategy="ardl_screen",
                lag_spec_strategy="full_product",
                top_n=3,
            ),
            ffnn=replace(
                config.ffnn,
                seed_grid=tuple(args.paper_seeds),
            ),
        )
        result = VMDARDLFFNNExperiment(
            hard_config
        ).run_hard_ffnn_comparator(
            args.data,
            reporting_seeds=tuple(args.paper_seeds),
        )
        print("Hard-config VMD-ARDL-FFNN metrics:")
        print(result["final_metrics"].to_string(index=False))
    elif args.pipeline == "sensitivity":
        from .sensitivity_experiments import run_sensitivity_experiments

        result = run_sensitivity_experiments(
            args.data,
            config,
            seeds=tuple(args.paper_seeds),
        )
        print("Sensitivity metrics:")
        print(result["metrics"].to_string(index=False))
    elif args.pipeline == "vmd-ardl-select-order":
        # Override the target lag strategy in the config to statsmodels ardl_select_order
        config = replace(
            config,
            ardl=replace(config.ardl, target_lag_strategy="ardl_select_order")
        )
        experiment = VMDARDLFFNNExperiment(config)
        result = experiment.run(args.data)
        print("Best model rows:")
        print(result["best_component_models"].to_string(index=False))
        print("\nFinal metrics:")
        print(result["final_metrics"].to_string(index=False))
    elif args.pipeline == "vmd-ardl-joint-select":
        # Override the target lag strategy in the config to statsmodels ardl_select_order_joint
        config = replace(
            config,
            ardl=replace(config.ardl, target_lag_strategy="ardl_select_order_joint")
        )
        experiment = VMDARDLFFNNExperiment(config)
        result = experiment.run(args.data)
        print("Best model rows:")
        print(result["best_component_models"].to_string(index=False))
        print("\nFinal metrics:")
        print(result["final_metrics"].to_string(index=False))
    else:
        result = experiment.run(args.data)
        print("Best model rows:")
        print(result["best_component_models"].to_string(index=False))
        print("\nFinal metrics:")
        print(result["final_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
