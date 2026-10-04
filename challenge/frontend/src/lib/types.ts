export type FeatureValue = { feature:string; value:number };
export type Window = { date:string; history_dates:string[]; features:string[]; values:number[][]; default_values:number[][] };
export type Snapshot = {
  filename:string; completed:string[]; features:string[]; labels:Record<string,string>;
  history:{date:string;arrivals:number|null;rainfall_mm:number|null;temp_mean_c:number|null}[];
  preview:Record<string,string|number|null>[]; window:Window|null;
  reports: {
    dataset?: {rows:number;columns:number;unique_months:number;start:string;end:string;header_rows_skipped:number;duplicate_records:number;missing_months:string[];missing:Record<string,number>;dtypes:{column:string;type:string;missing:number}[]};
    clean?: {rows:number;duplicates_removed:number;missing_before:Record<string,number>;missing_after:Record<string,number>;fill_log:{date:string;feature:string;value:number;method:string}[];outliers:{date:string;arrivals:number}[];iqr:{q1:number;q3:number;lower:number;upper:number};outlier_policy:string;encoding:string};
    features?: {results:{feature:string;rho:number|null;p_value:number|null;n:number;passed:boolean}[];vif_log:{iteration:number;dropped:string;vif:number|null;infinite:boolean}[];final_vif:Record<string,number>;selected:string[];fitting_start:string;fitting_end:string;fitting_rows:number;config:{train_ratio:number;validation_ratio:number;lookback:number;seed:number}};
    prepare?: {config:{train_ratio:number;validation_ratio:number;lookback:number};partitions:{name:string;start:string;end:string;rows:number;windows:number;shape:number[];warmup_rows:number;unusable_targets:number;first_target:string;last_target:string}[];scaler_ranges:{feature:string;min:number;max:number}[];policy:string};
    train?: {best_params:{units:number;dropout:number;batch_size:number};validation_mse:number;search:string;max_epochs:number;seed:number;trials:{units:number;dropout:number;batch_size:number;validation_mse:number;epochs_run:number;best_epoch:number}[];history:{epoch:number;training:number;validation:number}[];policy:string};
    evaluate?: {metrics:{model:string;MAE:number;RMSE:number;MAPE:number|null;R2:number|null;MAPE_n:number;n:number}[];predictions:{date:string;actual:number;lstm:number;naive:number;seasonal:number;error:number}[];lowest_mae:string;excluded_months:number;negative_predictions:number;policy:string};
    explain?: {method:string;samples:number;background_windows:number;background_centroids:number;masks:number;lookback:number;importance:FeatureValue[];top_feature:string;local:{date:string;base:number;prediction:number;residual:number;contributions:FeatureValue[]}[];dependence:{value:number;shap:number;date:string}[];policy:string};
    forecast?: {arrivals:number;date:string;negative:boolean;outside_training_range:{feature:string;cells:number}[];lookback:number;readings:number[][];policy:string};
  };
};
export type Job = {id:string;kind:string;status:string;progress:number;message:string;error:string|null};
