import pandas as pd
import numpy as np
import os
import yfinance as yf
from datetime import datetime, timedelta

pd.set_option('future.no_silent_downcasting', True)

from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

ETF_SYMBOLS = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XHB","XRT","XSD","SPY","QQQ", "BND","TLT","LQD","IEF","AGG","EMB","SHV","HYG", "GLD","SLV","USO","DBA","DBC", "EEM","EFA", "RWR","VNQ","VIG","VBINX","IWM","VTI","ACWI","SDY","VUG","VTV"]

PERIOD_1M = 21
PERIOD_2M = 42
PERIOD_3M = 63
PERIOD_6M = 126
PERIOD_9M = 189
PERIOD_12M = 252
ALL_PERIODS = {
    '1M': PERIOD_1M, '2M': PERIOD_2M, '3M': PERIOD_3M,
    '6M': PERIOD_6M, '9M': PERIOD_9M, '12M': PERIOD_12M
}

def read_columns_and_clean(file_path, sheet_name, columns_to_read=None, index_col=0, skiprows=None):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=index_col, skiprows=skiprows)
        if df.index.dtype != 'datetime64[ns]':
            df.index = pd.to_datetime(df.index)
        if columns_to_read:
            valid_columns = [col for col in columns_to_read if col in df.columns]
            missing_columns = [col for col in columns_to_read if col not in df.columns]
            if missing_columns:
                print(f"Warning: Columns {missing_columns} not found in sheet {sheet_name} of {file_path}. They will be skipped.")
            df = df[valid_columns]

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise
    except Exception as e:
        print(f"Error reading sheet {sheet_name} from {file_path}: {e}")
        raise

def convert_float_columns_to_datetime_user_provided(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            try:
                df[col] = pd.to_datetime(df[col].astype(str).str.split('.').str[0], format='%Y%m%d')
            except ValueError: 
                try:
                    df[col] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df[col], 'D')
                except Exception as e_inner:
                    print(f"Warning: Could not convert column {col} to datetime: {e_inner}. Leaving as is.")
    return df

def get_etf_universe_affinity_propagation(etf_prices_full_df, start_date_affinity='2010-01-01', end_date_affinity='2014-12-31', ap_preference=-30, ap_damping=0.7, random_state=48, fallback_tickers=None):
    """
    """
    try:
        etf_returns_full_df = etf_prices_full_df.pct_change()
        
        etf_subset_returns = etf_returns_full_df.loc[start_date_affinity:end_date_affinity]
        etf_subset_returns.dropna(axis=1, how='all', inplace=True)
        etf_subset_returns.dropna(axis=0, how='any', inplace=True)
        
        if etf_subset_returns.shape[1] < 2:
            print(f"Warning: Not enough ETF data ({etf_subset_returns.shape[1]} ETFs with full data) in the period {start_date_affinity}-{end_date_affinity} for Affinity Propagation. Using fallback.")
            return fallback_tickers if fallback_tickers else []

        etf_mean = etf_subset_returns.mean(axis=0)
        etf_std = etf_subset_returns.std(axis=0)
        

        etf_features_basic = pd.DataFrame({'Mean': etf_mean, 'StdDev': etf_std})

        corr_subset_returns = etf_subset_returns[etf_features_basic.index].corr()
        
        corr_subset_returns_aligned = corr_subset_returns.reindex(index=etf_features_basic.index, columns=etf_features_basic.index).add_suffix('_corr')
        
        etf_features = pd.concat([etf_features_basic, corr_subset_returns_aligned], axis=1)
        
        etf_features.dropna(axis=0, how='any', inplace=True)
        etf_features.fillna(0, inplace=True)

        if etf_features.empty:
            print("Warning: ETF features DataFrame is empty after processing. Using fallback.")
            return fallback_tickers if fallback_tickers else []

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(etf_features)

        if scaled_features.shape[0] == 0:
            print("Warning: No features to cluster after scaling. Using fallback.")
            return fallback_tickers if fallback_tickers else []

        af_model = AffinityPropagation(preference=ap_preference, damping=ap_damping, random_state=random_state, max_iter=500, convergence_iter=50)
        af_model.fit(scaled_features)
        
        cluster_centers_indices = af_model.cluster_centers_indices_
        
        if cluster_centers_indices is not None and len(cluster_centers_indices) > 0:
            selected_tickers_ap = etf_features.index[cluster_centers_indices].tolist()
            return selected_tickers_ap
        else:
            return fallback_tickers if fallback_tickers else []

    except Exception as e:
        import traceback
        traceback.print_exc()
        return fallback_tickers if fallback_tickers else []

def load_all_data(data_path_prefix=""):
    etf_summary_path = os.path.join(data_path_prefix, "ETF_Summary.xlsx")
    us_econ_data_path = os.path.join(data_path_prefix, "US_Economic_Data.xlsx")
    us_rates_path = os.path.join(data_path_prefix, "US Rates.xlsx")


    fallback_chosen_tickers = ['BND', 'EMB', 'SHV', 'GLD', 'DBC', 'IWM'] 

    start_date_yfinance = '2000-01-01'
    end_date_yfinance = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    cache_dir = os.path.join(data_path_prefix, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    adj_close_data = {}
    volume_data = {}
    
    standard_cols_order = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']

    for ticker in ETF_SYMBOLS:
        ticker_filepath = os.path.join(cache_dir, f"{ticker}.csv")
        data_df_for_ticker = None
        
        try:
            if os.path.exists(ticker_filepath):
                with open(ticker_filepath, 'r') as f:
                    header_line = f.readline().strip()
                
                if not header_line:
                    os.remove(ticker_filepath)
                    raise FileNotFoundError("Corrupt cache file removed, retrying download.")

                column_names_from_file = header_line.split(',')
                
                data_df_for_ticker = pd.read_csv(ticker_filepath, skiprows=3, header=None, index_col=0, parse_dates=True)
                
                if not data_df_for_ticker.empty:
                    if len(data_df_for_ticker.columns) == len(column_names_from_file):
                        data_df_for_ticker.columns = column_names_from_file
                    elif len(column_names_from_file) == 6 and len(data_df_for_ticker.columns) == 5:
                        data_df_for_ticker.columns = column_names_from_file[:-1]
                    else:
                        os.remove(ticker_filepath)
                        raise FileNotFoundError("Cache file column mismatch (unhandled case), retrying download.")
                elif column_names_from_file:
                   data_df_for_ticker = pd.DataFrame(columns=column_names_from_file, index=pd.to_datetime([]))
                   data_df_for_ticker.index.name = 'Date'


            if not os.path.exists(ticker_filepath):
                raw_yf_data = yf.download(ticker, start=start_date_yfinance, end=end_date_yfinance, progress=False, timeout=10)
                
                if raw_yf_data.empty:
                    print(f"Warning: No data downloaded for {ticker}. Skipping.")
                    continue

                data_to_save = raw_yf_data.copy()
                if 'Close' in data_to_save.columns:
                    data_to_save['Price'] = data_to_save['Close']
                else:
                    print(f"Warning: Neither 'Adj Close' nor 'Close' found for {ticker}. Cannot create 'Price' column. Skipping save.")
                    continue

                actual_cols_present_for_saving = [col for col in standard_cols_order if col in data_to_save.columns]
                data_to_save_final = data_to_save[actual_cols_present_for_saving]

                if data_to_save_final.empty and not actual_cols_present_for_saving :
                    print(f"Warning: No standard columns found to save for {ticker}. Skipping save.")
                    continue


                with open(ticker_filepath, 'w') as f:
                    f.write(','.join(actual_cols_present_for_saving) + '\n')
                    if actual_cols_present_for_saving:
                            f.write(f"Ticker,{','.join([ticker] * (len(actual_cols_present_for_saving)-1))}\n")
                    else:
                            f.write("Ticker\n")

                    if actual_cols_present_for_saving:
                            f.write("Date" + ("," * (len(actual_cols_present_for_saving) - 1) if len(actual_cols_present_for_saving) > 0 else "") + "\n")
                    else:
                            f.write("Date\n")

                data_to_save_final.to_csv(ticker_filepath, mode='a', header=False, index=True)
                data_df_for_ticker = data_to_save_final
            
            if data_df_for_ticker is None or data_df_for_ticker.empty: 
                print(f"Warning: No data available for {ticker} after download/cache attempt.")
                continue
            
            if 'Price' in data_df_for_ticker.columns:
                adj_close_data[ticker] = data_df_for_ticker['Price']
            elif 'Close' in data_df_for_ticker.columns:
                adj_close_data[ticker] = data_df_for_ticker['Close']
                print(f"Note: Using 'Close' price for {ticker} as 'Price' column was not generated/loaded.")
            else:
                print(f"Warning: Crucial 'Price' or 'Close' column not found for {ticker}. Skipping ticker for price data.")
            
            if 'Volume' in data_df_for_ticker.columns:
                volume_data[ticker] = data_df_for_ticker['Volume']
            else:
                print(f"Warning: 'Volume' column not found for {ticker}.")
                if ticker in adj_close_data:
                    volume_data[ticker] = pd.Series(np.nan, index=adj_close_data[ticker].index, name='Volume')


        except FileNotFoundError:

            if os.path.exists(ticker_filepath):
                print(f"Error: File {ticker_filepath} not found unexpectedly.")

            if not os.path.exists(ticker_filepath):
                data_df_for_ticker = None 

                raw_yf_data = yf.download(ticker, start=start_date_yfinance, end=end_date_yfinance, progress=False, timeout=10)
                
                if raw_yf_data.empty:
                    print(f"Warning: No data downloaded for {ticker} on retry. Skipping.")
                    continue

                data_to_save = raw_yf_data.copy()
                if 'Adj Close' in data_to_save.columns:
                    data_to_save.rename(columns={'Adj Close': 'Price'}, inplace=True)
                elif 'Close' in data_to_save.columns:
                    data_to_save['Price'] = data_to_save['Close']
                else:
                    print(f"Warning: Neither 'Adj Close' nor 'Close' found for {ticker} on retry. Cannot create 'Price'. Skipping save.")
                    continue

                actual_cols_present_for_saving = [col for col in standard_cols_order if col in data_to_save.columns]
                data_to_save_final = data_to_save[actual_cols_present_for_saving]

                if data_to_save_final.empty and not actual_cols_present_for_saving :
                    print(f"Warning: No standard columns found to save for {ticker} on retry. Skipping save.")
                    continue

                with open(ticker_filepath, 'w') as f:
                    f.write(','.join(actual_cols_present_for_saving) + '\n')
                    if actual_cols_present_for_saving:
                         f.write(f"Ticker,{','.join([ticker] * (len(actual_cols_present_for_saving)-1))}\n")
                    else:
                         f.write("Ticker\n")
                    if actual_cols_present_for_saving:
                         f.write("Date" + ("," * (len(actual_cols_present_for_saving) - 1) if len(actual_cols_present_for_saving) > 0 else "") + "\n")
                    else:
                         f.write("Date\n")
                    
                data_to_save_final.to_csv(ticker_filepath, mode='a', header=False, index=True)
                data_df_for_ticker = data_to_save_final

                if 'Price' in data_df_for_ticker.columns:
                    adj_close_data[ticker] = data_df_for_ticker['Price']
                elif 'Close' in data_df_for_ticker.columns:
                    adj_close_data[ticker] = data_df_for_ticker['Close']
                else:
                    print(f"Warning: Crucial 'Price' or 'Close' column not found for {ticker} on retry. Skipping for price data.")
                
                if 'Volume' in data_df_for_ticker.columns:
                    volume_data[ticker] = data_df_for_ticker['Volume']
                else:
                    print(f"Warning: 'Volume' column not found for {ticker} on retry.")
                    if ticker in adj_close_data:
                        volume_data[ticker] = pd.Series(np.nan, index=adj_close_data[ticker].index, name='Volume')

        except Exception as e: 
            print(f"General Warning: Could not download or load data for {ticker}: {e}. File: {ticker_filepath}")


    etf_prices_full_df = pd.DataFrame(adj_close_data)
    etf_volume_full_df = pd.DataFrame(volume_data)
    
    etf_prices_full_df.ffill(inplace=True); etf_prices_full_df.bfill(inplace=True)
    etf_volume_full_df.ffill(inplace=True); etf_volume_full_df.bfill(inplace=True)

    adjusted_start_date = '2010-01-01'
    etf_prices_full_df = etf_prices_full_df.loc[adjusted_start_date:]
    etf_volume_full_df = etf_volume_full_df.loc[adjusted_start_date:]

    selected_etf_tickers = get_etf_universe_affinity_propagation(
        etf_prices_full_df.copy(),
        fallback_tickers=fallback_chosen_tickers
    )
    if not selected_etf_tickers:
        print("Critical Error: No ETF tickers selected by Affinity Propagation or fallback. Aborting.")
        return {'chosen_etf_prices': pd.DataFrame(), 'combined_features': pd.DataFrame(), 'selected_tickers': []}
    
    chosen_tickers_from_selection = selected_etf_tickers 

    valid_price_cols = [col for col in chosen_tickers_from_selection if col in etf_prices_full_df.columns]
    chosen_etf_prices_df = etf_prices_full_df[valid_price_cols].copy()

    valid_volume_cols = [col for col in chosen_tickers_from_selection if col in etf_volume_full_df.columns]
    chosen_etf_volume_df = etf_volume_full_df[valid_volume_cols].copy()
    
    if chosen_etf_prices_df.empty:
        raise ValueError("Chosen ETF prices DataFrame is empty after selection. This might indicate issues with the selected tickers or data fetching.")

    momentum_features = {}
    for N_str, N_val in ALL_PERIODS.items():
        momentum_features[f'M{N_str}_return'] = chosen_etf_prices_df / chosen_etf_prices_df.shift(N_val) - 1
    
    m1_forward_return_df = chosen_etf_prices_df.shift(-PERIOD_1M) / chosen_etf_prices_df - 1
    m1_forward_return_df.rename(columns=lambda c: f"{c}_M1_fwd_rtn", inplace=True)

    skip_rows_etf_summary = [0,1,2,3,4,6] 
    try:
        market_cap_df_raw = read_columns_and_clean(etf_summary_path, "MARKET_CAP", chosen_tickers_from_selection, skiprows=skip_rows_etf_summary)
        put_call_df_raw = read_columns_and_clean(etf_summary_path, "PUT_CALL", chosen_tickers_from_selection, skiprows=skip_rows_etf_summary)
        short_int_df_raw = read_columns_and_clean(etf_summary_path, "SHORT_INT", chosen_tickers_from_selection, skiprows=skip_rows_etf_summary)
        iv_1m_atm_df_raw = read_columns_and_clean(etf_summary_path, "1M_50D", chosen_tickers_from_selection, skiprows=skip_rows_etf_summary)

        base_index = chosen_etf_prices_df.index
        market_cap_df = market_cap_df_raw.reindex(base_index).ffill()
        put_call_df = put_call_df_raw.reindex(base_index).ffill()
        short_int_df = short_int_df_raw.reindex(base_index).ffill()
        iv_1m_atm_df = iv_1m_atm_df_raw.reindex(base_index).ffill()
    except Exception as e:
        print(f"Warning: Could not load supplemental features from {etf_summary_path} for selected tickers: {e}. Using placeholders.")
        placeholder_cols = [col for col in chosen_tickers_from_selection if col in chosen_etf_prices_df.columns]
        market_cap_df = pd.DataFrame(np.nan, index=chosen_etf_prices_df.index, columns=placeholder_cols)
        put_call_df = pd.DataFrame(np.nan, index=chosen_etf_prices_df.index, columns=placeholder_cols)
        short_int_df = pd.DataFrame(np.nan, index=chosen_etf_prices_df.index, columns=placeholder_cols)
        iv_1m_atm_df = pd.DataFrame(np.nan, index=chosen_etf_prices_df.index, columns=placeholder_cols)


    skip_rows_econ = [0,1,2,3,5,6]
    econ_vars_to_transform = ["GDPC1", "PCEPI", "INDPRO", "CPIAUCSL", "PAYEMS", "UMCSENT", "UNRATE", "HOUST", "PERMIT", "TOTALSA", "NAPM", "PPIACO", "PCECTPI", "DSPIC96"]

    try:
        actual_release_df_full = read_columns_and_clean(us_econ_data_path, "Actual", index_col=0, skiprows=skip_rows_econ)
        actual_release_df = actual_release_df_full[econ_vars_to_transform].copy() if not actual_release_df_full.empty else pd.DataFrame()

        release_date_df_raw_full = read_columns_and_clean(us_econ_data_path, "ReleaseDate", index_col=0, skiprows=skip_rows_econ)
        release_date_df_raw = release_date_df_raw_full[econ_vars_to_transform].copy() if not release_date_df_raw_full.empty else pd.DataFrame()

        release_date_df = convert_float_columns_to_datetime_user_provided(release_date_df_raw.copy())
        
        adjusted_actual_release = pd.DataFrame(index=actual_release_df.index, columns=actual_release_df.columns)
        
        release_date_df_aligned = release_date_df.reindex(columns=actual_release_df.columns).ffill().bfill()

        for col_name in actual_release_df.columns:
            if col_name in release_date_df_aligned.columns:
                for original_date, value in actual_release_df[col_name].items():
                    if pd.notnull(value):
                        actual_release_dt = release_date_df_aligned.loc[original_date, col_name]
                        if pd.notnull(actual_release_dt) and isinstance(actual_release_dt, pd.Timestamp):
                            adjusted_actual_release.loc[actual_release_dt, col_name] = value
            else:
                print(f"Warning: Column {col_name} not found in ReleaseDate sheet. Skipping this economic indicator.")

        adjusted_actual_release.sort_index(axis=0, ascending=True, inplace=True)
        adjusted_actual_release.dropna(how='all', axis=0, inplace=True)
        adjusted_actual_release = adjusted_actual_release[adjusted_actual_release.index.notnull()]
        
        adjusted_actual_release = adjusted_actual_release.ffill(axis=0).bfill(axis=0)

        econ_cols_present = [col for col in econ_vars_to_transform if col in adjusted_actual_release.columns]
        
        adjusted_actual_release_change = adjusted_actual_release[econ_cols_present].diff(periods=PERIOD_3M)
        adjusted_actual_release_change.columns = [f"{col}_3M_diff" for col in adjusted_actual_release_change.columns]
        
        processed_econ_df = pd.concat([adjusted_actual_release, adjusted_actual_release_change], axis=1)
    except Exception as e:
        print(f"Error processing economic data from {us_econ_data_path}: {e}. Using empty DataFrame.")
        processed_econ_df = pd.DataFrame(index=chosen_etf_prices_df.index)

    try:
        us_rates_df_full = pd.read_excel(us_rates_path, index_col=0)
        base_rate_columns = ['DGS10', 'DGS2', 'DGS1', 'DGS3MO']
        actual_base_rate_columns = [col for col in base_rate_columns if col in us_rates_df_full.columns]
        us_rates_df = us_rates_df_full[actual_base_rate_columns].copy()

        us_rates_df.index = pd.to_datetime(us_rates_df.index)
        if 'DGS10' in us_rates_df.columns and 'DGS2' in us_rates_df.columns:
            us_rates_df['Y10_Y2_Spread'] = us_rates_df['DGS10'] - us_rates_df['DGS2']
        
        us_rates_df = us_rates_df.ffill().bfill()

        rate_cols_to_transform = ['DGS10', 'DGS2', 'DGS1', 'DGS3MO', 'Y10_Y2_Spread'] 
        present_rate_cols = [col for col in rate_cols_to_transform if col in us_rates_df.columns]
        
        us_rates_change_3m = us_rates_df[present_rate_cols].diff(periods=PERIOD_3M)
        us_rates_change_3m.columns = [f"{col}_3M_diff" for col in us_rates_change_3m.columns]
        
        us_rates_std_3m = us_rates_df[present_rate_cols].rolling(window=PERIOD_3M).std()
        us_rates_std_3m.columns = [f"{col}_3M_std" for col in us_rates_std_3m.columns]
        
        processed_rates_df = pd.concat([us_rates_df, us_rates_change_3m, us_rates_std_3m], axis=1)
    except Exception as e:
        print(f"Error processing rates data from {us_rates_path}: {e}. Using empty DataFrame.")
        processed_rates_df = pd.DataFrame(index=chosen_etf_prices_df.index)

    all_feature_dfs_stacked = []
    for name, df in momentum_features.items():
        if not df.empty:
            try:
                stacked_df = df.stack().reset_index(name=name)
                stacked_df.columns = ['Date', 'Ticker', name]
                stacked_df.set_index(['Date', 'Ticker'], inplace=True)
                all_feature_dfs_stacked.append(stacked_df)
            except ValueError as e:
                raise


    
    if not m1_forward_return_df.empty:
        stacked_fwd_returns = m1_forward_return_df.stack().reset_index(name='M1_fwd_rtn'); stacked_fwd_returns.columns = ['Date', 'Ticker', 'M1_fwd_rtn']; stacked_fwd_returns.set_index(['Date', 'Ticker'], inplace=True); all_feature_dfs_stacked.append(stacked_fwd_returns)
    
    if not chosen_etf_volume_df.empty:
        stacked_volume = chosen_etf_volume_df.stack().reset_index(name='Volume'); stacked_volume.columns = ['Date', 'Ticker', 'Volume']; stacked_volume.set_index(['Date', 'Ticker'], inplace=True); all_feature_dfs_stacked.append(stacked_volume)
    
    supplemental_dfs = {'MarketCap': market_cap_df, 'PutCallRatio': put_call_df, 'ShortInterest': short_int_df, 'IV_1M_ATM': iv_1m_atm_df}
    for name, df in supplemental_dfs.items():
        if not df.empty and not df.stack().empty:
            stacked_df = df.stack().reset_index(name=name); stacked_df.columns = ['Date', 'Ticker', name]; stacked_df.set_index(['Date', 'Ticker'], inplace=True); all_feature_dfs_stacked.append(stacked_df)
            
    if not all_feature_dfs_stacked: 
        print("Warning: No ETF-specific features available to combine. Combined_df will only have econ/rates.")

        if chosen_etf_prices_df.empty:
            raise ValueError("No ETF-specific features and chosen_etf_prices_df is empty. Cannot proceed.")

        base_multi_index = pd.MultiIndex.from_product([chosen_etf_prices_df.index, chosen_etf_prices_df.columns], names=['Date', 'Ticker'])
        combined_df = pd.DataFrame(index=base_multi_index)

    else:
        combined_df = pd.concat(all_feature_dfs_stacked, axis=1)

    combined_df.reset_index(inplace=True)
    

    if not processed_econ_df.empty:
        combined_df = pd.merge(combined_df, processed_econ_df, left_on='Date', right_index=True, how='left')
    if not processed_rates_df.empty:
        combined_df = pd.merge(combined_df, processed_rates_df, left_on='Date', right_index=True, how='left')
        
    combined_df.set_index(['Date', 'Ticker'], inplace=True)
    
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True)

    if 'M1_fwd_rtn' in combined_df.columns:
        combined_df.dropna(subset=['M1_fwd_rtn'], inplace=True)
    
    combined_df.dropna(how='all', axis=1, inplace=True)

    malformed_cols = [col for col in combined_df.columns if 'Unnamed:' in str(col)]
    if malformed_cols:
        print(f"Dropping malformed columns from combined_features in data_loader: {malformed_cols}")
        combined_df = combined_df.drop(columns=malformed_cols)

    return {
        'chosen_etf_prices': chosen_etf_prices_df, 
        'combined_features': combined_df,
        'selected_tickers': chosen_tickers_from_selection
    }


if __name__ == '__main__':

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_path)
    data_path_prefix = os.path.join(project_root, 'data')

    data_path_prefix = ""

    try:
        all_loaded_data = load_all_data(data_path_prefix=data_path_prefix)


    except FileNotFoundError as e:
        print(f"A required data file was not found: {e}. Please ensure ETF_Summary.xlsx, US_Economic_Data.xlsx, and US Rates.xlsx are accessible.")
    except ValueError as e:
        print(f"ValueError during data loading: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in the main execution block: {e}")
        import traceback
        traceback.print_exc()
