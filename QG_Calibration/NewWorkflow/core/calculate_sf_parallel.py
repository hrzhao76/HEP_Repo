# This script is used in the end of make_histogram.py for plotting with multicores. 
from .Calculate_SF import  * 
from .utils import check_outputpath
from concurrent.futures import ProcessPoolExecutor
import functools 

def calculate_sf_parallel(plot_tuple:dict, output_path, period='ADE'):
    hists_MC = plot_tuple[1]['MC']
    hists_Data = plot_tuple[1]['Data']
    output_path = check_outputpath(output_path)

    if plot_tuple[0] == 'event_weight':
        reweighting_var = 'none'
        weight_option = plot_tuple[0]
    else:
        reweighting_var = '_'.join(str.split(plot_tuple[0], '_')[:2])
        weight_option = '_'.join(str.split(plot_tuple[0], '_')[2:])
    
    HistMap_MC_unumpy = convert_histdict2unimpy(hists_MC)
    HistMap_Data_unumpy = convert_histdict2unimpy(hists_Data)

    #### Draw pt spectrum
    Plot_Pt_Spectrum(HistMap_MC_unumpy, HistMap_Data_unumpy, output_path, reweighting_var, weight_option)


    Extraction_Results = Extract(HistMap_MC_unumpy, HistMap_Data_unumpy)
    # joblib.dump(Extraction_Results, output_path / f"{reweighting_var}_Extraction_Results.pkl")
    #### Draw ROC plot 
    Plot_ROC(Extraction_Results, output_path, period, reweighting_var, reweighting_option=weight_option)

    WPs = [0.5, 0.6, 0.7, 0.8]
    SFs = {}
    WP_cut = {}

    for var in label_var:
        SFs[var] = {}
        WP_cut[var] = {}
        for l_pt in label_ptrange[:-1]:
            Extraction_var_pt =  Extraction_Results[var][l_pt]
            #### Draw Forward vs Central plots 
            Plot_ForwardCentral_MCvsData(pt = l_pt, var= var, output_path= output_path, 
                                period= period, reweighting_var = reweighting_var,
                                reweighting_option= weight_option,
                                Forward_MC= Extraction_var_pt['Forward_MC'], 
                                Central_MC= Extraction_var_pt['Central_MC'],
                                Forward_Data= Extraction_var_pt['Forward_Data'], 
                                Central_Data= Extraction_var_pt['Central_Data'],
                                if_norm=False, show_yields=True)

            Plot_ForwardCentral_MCvsData(pt = l_pt, var = var, output_path = output_path, 
                                period = period, reweighting_var = reweighting_var,
                                reweighting_option= weight_option,
                                Forward_MC= Normalize_unumpy(Extraction_var_pt['Forward_MC']), 
                                Central_MC= Normalize_unumpy(Extraction_var_pt['Central_MC']),
                                Forward_Data= Normalize_unumpy(Extraction_var_pt['Forward_Data']), 
                                Central_Data= Normalize_unumpy(Extraction_var_pt['Central_Data']),
                                if_norm=True, show_yields=False)

            Plot_Parton_ForwardvsCentral(pt = l_pt, var = var, output_path = output_path,
                                period = period, reweighting_var = reweighting_var,
                                reweighting_option = weight_option, 
                                p_Forward_Quark = Normalize_unumpy(Extraction_var_pt['Forward_Quark']),  
                                p_Central_Quark = Normalize_unumpy(Extraction_var_pt['Central_Quark']), 
                                p_Forward_Gluon = Normalize_unumpy(Extraction_var_pt['Forward_Gluon']), 
                                p_Central_Gluon = Normalize_unumpy(Extraction_var_pt['Central_Gluon'])
                                )

            #### Draw extraction plots 
            Plot_Extracted_unumpy(pt = l_pt, var= var, output_path= output_path, 
                                    period= period, reweighting_var = reweighting_var,
                                    reweighting_factor= weight_option,
                                    p_Quark=Extraction_var_pt['p_Quark'], p_Gluon=Extraction_var_pt['p_Gluon'],
                                    extract_p_Quark = Extraction_var_pt['extract_p_Quark_MC'], extract_p_Gluon = Extraction_var_pt['extract_p_Gluon_MC'],
                                    extract_p_Quark_Data = Extraction_var_pt['extract_p_Quark_Data'], extract_p_Gluon_Data = Extraction_var_pt['extract_p_Gluon_Data'],
                                    show_yields=True, 
                                    n_Forward_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_MC'])), 
                                    n_Central_MC = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_MC'])), 
                                    variances_Forward_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Forward_MC'])**2),
                                    variances_Central_MC= np.sum(unumpy.std_devs(Extraction_var_pt['Central_MC'])**2), 
                                    n_Forward_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Forward_Data'])), 
                                    n_Central_Data = np.sum(unumpy.nominal_values(Extraction_var_pt['Central_Data'])),
                                    variances_Forward_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Forward_Data'])**2),
                                    variances_Central_Data=np.sum(unumpy.std_devs(Extraction_var_pt['Central_Data'])**2)
                                    )
        #### Draw working points 
        for WP in WPs:
            SFs[var][WP] = {}
            WP_cut[var][WP] = {}
            quark_effs_at_pt = []
            gluon_rejs_at_pt = []
            quark_effs_data_at_pt = []
            gluon_rejs_data_at_pt = []
            for ii, l_pt in enumerate(label_ptrange[:-1]):
                extract_p_Quark_MC =  Extraction_Results[var][l_pt]['extract_p_Quark_MC']
                extract_p_Gluon_MC =  Extraction_Results[var][l_pt]['extract_p_Gluon_MC']
                extract_p_Quark_Data =  Extraction_Results[var][l_pt]['extract_p_Quark_Data']
                extract_p_Gluon_Data =  Extraction_Results[var][l_pt]['extract_p_Gluon_Data']

                extract_p_Quark_cum_sum = np.cumsum(unumpy.nominal_values(extract_p_Quark_MC))
                cut = np.where(extract_p_Quark_cum_sum >= WP)[0][0]+1
                
                
                quark_effs_at_pt.append(np.sum(extract_p_Quark_MC[:cut])) 
                gluon_rejs_at_pt.append(np.sum(extract_p_Gluon_MC[cut:]))
                quark_effs_data_at_pt.append(np.sum(extract_p_Quark_Data[:cut]))
                gluon_rejs_data_at_pt.append(np.sum(extract_p_Gluon_Data[cut:]))
                WP_cut[var][WP][l_pt] = {
                    'idx' : cut,
                    'value' : HistBins[var][cut],
                }

            SF_quark, SF_gluon = Plot_WP(WP = WP, var= var, output_path= output_path, 
                    period= period, reweighting_var = reweighting_var,
                    reweighting_factor= weight_option,
                    quark_effs= quark_effs_at_pt, gluon_rejs = gluon_rejs_at_pt,
                    quark_effs_data=quark_effs_data_at_pt, gluon_rejs_data = gluon_rejs_data_at_pt)
            SFs[var][WP]["Quark"] = SF_quark
            SFs[var][WP]["Gluon"] = SF_gluon

        WriteSFtoPickle(var = var,Hist_SFs = SFs, output_path=output_path, period=period, 
                        reweighting_var = reweighting_var, reweighting_factor= weight_option)
        WriteWPcuttoPickle(var = var,WP_cuts= WP_cut, output_path=output_path, period=period, 
                        reweighting_var = reweighting_var, reweighting_factor= weight_option)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This python script does the MC Closure test. ')
    parser.add_argument('--path-mc', help='The path to the merged MC histogram file(.pkl file).',\
        default='/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/nominal/MC_merged_hist.pkl')
    parser.add_argument('--path-data', help='The path to the merged Data histogram file(.pkl file).',\
        default='/global/cfs/projectdirs/atlas/hrzhao/HEP_Repo/QG_Calibration/NewWorkflow/trained_lightGBM_new/nominal/Data_merged_hist.pkl')
    parser.add_argument('--period', help='The MC16 period', choices=['A', 'D', 'E', "ADE"],\
        default='ADE')
    parser.add_argument('--output-path', help='Output path',\
        default='./test_calculate_sf_parallel')

    args = parser.parse_args()
    
    MC_merged_hist_path = Path(args.path_mc)
    Data_merged_hist_path = Path(args.path_data)
    period = Path(args.period)
    output_path = Path(args.output_path)

    MC_merged_hist = joblib.load(MC_merged_hist_path)
    Data_merged_hist = joblib.load(Data_merged_hist_path)

    plot_dict = {}
    for key in [*MC_merged_hist.keys()]:
        plot_dict[key]={
            "MC":MC_merged_hist[key],
            "Data":Data_merged_hist[key],
        }
    plot_tuple_list = [*plot_dict.items()]

    n_worker_plots = len(plot_tuple_list)
    calculate_sf_parallel_mod = functools.partial(calculate_sf_parallel, period = period, output_path=output_path)
    with ProcessPoolExecutor(max_workers=n_worker_plots) as executor:
        executor.map(calculate_sf_parallel_mod, plot_tuple_list)
    pass