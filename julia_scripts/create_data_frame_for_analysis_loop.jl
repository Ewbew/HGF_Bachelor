using ActionModels
using HierarchicalGaussianFiltering
using Plots
using StatsPlots
using Distributions
using CSV
using DataFrames
using FileIO
using JLD2
using MCMCChains
using AxisArrays

#=
This file is for creating the a unique dataframe for each type of parameter, 
namely the the evolution rates for x1, x2 and x3 and furthermore the gaussian action precision.
=#


data_path_loop = "/Users/ottosejrskildsantesson/Desktop/Bachelor/HGF_Bachelor/data/fitted_models/"
df = CSV.read(data_path * "unfiltered_pit_w_PDI_no_NA.csv", DataFrame);
data_path_data_frames = "/Users/ottosejrskildsantesson/Desktop/Bachelor/HGF_Bachelor/data/dataframes_of_parameters/"

parameters = ["GAP", "x1_e_rate", "x3_e_rate", "x2_e_rate"]
# GAP is gaussian_action_precision

for parameter in parameters
    dataframe = DataFrame(ID = [], session = [], mean = [], std = [], pdi_total = [], pdi_prop_yes = [], pdi_count = [], distress = [], preocc = [], belief = [])
    for ID in unique(df[!,:ID])
        for session in unique(df[!,:session])
            
            # filter the data for the current session and participant
            filtered_data = filter([:ID, :session] => (x, y) -> x == ID && y == session , df)

            # Skips the iteration, if the filtered data has no rows
            if size(filtered_data,1) == 0
                continue
            end


            filename = "participant_$(ID)_session_$(session).jld2"

            # Saving the file in the folder "data/fitted_models"
            fitted_model = load_object(data_path_loop * filename)

            # 1 = gaussian_action_precision
            # 2 = x1 evolution rate
            # 3 = x3 evolution rate
            # 4 = x2 evolution rate

            if parameter == "GAP"
                n = 1
            elseif parameter == "x1_e_rate"
                n = 2
            elseif parameter == "x3_e_rate"
                n = 3
            elseif parameter == "x2_e_rate"
                n = 4
            end
            
            mean = summarystats(fitted_model)[n, :mean]
            std = summarystats(fitted_model)[n, :std]
            pdi_total = filtered_data[1,:pdi_total]
            pdi_prop_yes = filtered_data[1,:pdi_prop_yes]
            pdi_count = filtered_data[1,:pdi_count]
            distress = filtered_data[1,:distress]
            preocc = filtered_data[1,:preocc]
            belief = filtered_data[1,:belief]
            confidence = filtered_data[1,:confidence]


            row = [ID, session, mean, std, pdi_total, pdi_prop_yes, pdi_count, distress, preocc, belief]

            # Add the row to the dataframe for the current parameter
            push!(dataframe, row)
        end
    end
    CSV.write(data_path_data_frames * "data_frame_$(parameter).csv", dataframe)
end
