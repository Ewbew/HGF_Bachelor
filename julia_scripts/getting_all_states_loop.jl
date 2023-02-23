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

data_path = "/Users/ottosejrskildsantesson/Desktop/Bachelor/HGF_Bachelor/data/"
df = CSV.read(data_path * "unfiltered_pit_w_PDI_no_NA.csv", DataFrame);

hgf = premade_hgf("JGET", verbose = false)
agent = premade_agent("hgf_gaussian_action", hgf)

params_for_all_agents = Dict(
    ("u", "x1", "value_coupling")       => 1, 
    ("x4", "evolution_rate")            => -15, # Taken it down by additional 3
    ("x4", "initial_mean")              => 1,
    ("x3", "x4", "volatility_coupling") => 1,
    ("u", "evolution_rate")             => 0,
    ("x4", "initial_precision")         => 1/0.1,
    ("x1", "initial_precision")         => 1/3, 
    ("x2", "initial_precision")         => 1/0.1,
    ## x1 initial mean
    ("x3", "initial_precision")         => 1/3, 
    ("x3", "initial_mean")              => log(25), # Doesn't seem to be logscaled in Nace's code
    ("x2", "initial_mean")              => 3,
    ("x1", "x2", "volatility_coupling") => 1,
    ("u", "x3", "volatility_coupling")  => 1,
    );



set_params!(agent, params_for_all_agents)


path_to_fitted_models = "/Users/ottosejrskildsantesson/Desktop/Bachelor/HGF_Bachelor/data/fitted_models/"

# Defining the function that gives a single input to the agent and then gets the surprise in return:
function set_unique_params(ID, session, filtered_unique_data)
    # Defining the unique filename for the fitted model, the parameters are to be extracted from
    filename_unique_fitted_model = "participant_$(ID)_session_$(session).jld2"

    # Loading in the object
    unique_fitted_model = load_object(path_to_fitted_models * filename_unique_fitted_model)
    
    # Overwriting the parameters, so they fit to this iteration of ID and session
    unique_agent_params = Dict(
    ("x1", "evolution_rate")            => summarystats(unique_fitted_model)[2, :mean],
    ("x2", "evolution_rate")            => summarystats(unique_fitted_model)[4, :mean], 
    ("x3", "evolution_rate")            => summarystats(unique_fitted_model)[3, :mean],
    "gaussian_action_precision"         => summarystats(unique_fitted_model)[1, :mean],
    ("x1", "initial_mean")              => filtered_unique_data[1,:outcome], 
    ); # The above rows were specified on grounds of the data_frame_loop.jl – the row number set corresponds to the different params

    # Overwriting the preexisiting parameters with the corresponding unique ones
    set_params!(agent, unique_agent_params)
end


df_get_history_extracts = DataFrame()

for ID in unique(df[!,:ID])
    for session in unique(df[!,:session])

        # Subsetting the dataset based on the ID and session number
        filtered_unique_data = filter([:ID, :session] => (x, y) -> x == ID && y == session , df)

        # If the size of the data is zero, then skip that iteration
        if size(filtered_unique_data,1) == 0
            continue
        end

        amount_of_combinations = amount_of_combinations + 1

        # Resetting the agent, removing all of the previous inputs
        reset!(agent)

        # Using the function defined above to the set the unique parameters of the agent
        set_unique_params(ID, session,filtered_unique_data)

        # Making sure that the rows are ordered accordingly to trials, from 1 - 240
        filtered_unique_data = filtered_unique_data[sortperm(filtered_unique_data[:,:trials]), :]

        try
            give_inputs!(agent, filtered_unique_data[!,:outcome])
            println("Success for: ID = $ID , session = $session")
        catch
            println("Fail for: ID = $ID , session = $session")

            continue
        end
        

        # println("ID = $ID , session = $session")
        push!(vector_check, ID)

        u_pred_precis = get_history(agent, "u")["prediction_precision"]
        u_pred_vol = get_history(agent, "u")["prediction_volatility"]
        u_val_pred_err = get_history(agent, "u")["value_prediction_error"][2:end] # Slicing to remove the last value, since otherwise the vector is not compatible with the data frame
        u_vol_pred_err = get_history(agent, "u")["volatility_prediction_error"][2:end]
        
        x1_pred_precis = get_history(agent, "x1")["prediction_precision"]
        x1_pred_vol = get_history(agent, "x1")["prediction_volatility"]
        x1_vol_pred_err = get_history(agent, "x1")["volatility_prediction_error"][2:end]
        x1_val_pred_err = get_history(agent, "x1")["value_prediction_error"][2:end]
        x1_post_mean = get_history(agent, "x1")["posterior_mean"][2:end]
        x1_post_precis = get_history(agent, "x1")["posterior_precision"][2:end]

        x2_pred_precis = get_history(agent, "x2")["prediction_precision"]
        x2_pred_vol = get_history(agent, "x2")["prediction_volatility"]
        # Volatility prediction error is not included for node x2, since the volatility is fixed (I think so)
        x2_val_pred_err = get_history(agent, "x2")["value_prediction_error"][2:end]
        x2_post_mean = get_history(agent, "x2")["posterior_mean"][2:end]
        x2_post_precis = get_history(agent, "x2")["posterior_precision"][2:end]

        x3_pred_precis = get_history(agent, "x3")["prediction_precision"]
        x3_pred_vol = get_history(agent, "x3")["prediction_volatility"]    
        x3_vol_pred_err = get_history(agent, "x3")["volatility_prediction_error"][2:end]
        x3_val_pred_err = get_history(agent, "x3")["value_prediction_error"][2:end]
        x3_post_mean = get_history(agent, "x3")["posterior_mean"][2:end]
        x3_post_precis = get_history(agent, "x3")["posterior_precision"][2:end]

        x4_pred_precis = get_history(agent, "x4")["prediction_precision"]
        x4_pred_vol = get_history(agent, "x4")["prediction_volatility"]
        # Volatility prediction error is not included for node x4 as well, since the volatility is fixed (I think so)
        x4_val_pred_err = get_history(agent, "x4")["value_prediction_error"][2:end]
        x4_post_mean = get_history(agent, "x4")["posterior_mean"][2:end]
        x4_post_precis =  get_history(agent, "x4")["posterior_precision"][2:end]


        # Constructing the data frame for all the new variables
        data_frame_get_history_extracts = DataFrame(
            outcome = get_history(agent, "u")["input_value"][2:end],
            ID = ID,
            session = session,
            trials = collect(1:size(filtered_unique_data,1)),

            u_pred_precis = u_pred_precis,
            u_pred_vol = u_pred_vol,
            u_val_pred_err = u_val_pred_err,
            u_vol_pred_err = u_vol_pred_err,

            x1_pred_precis = x1_pred_precis,
            x1_pred_vol = x1_pred_vol,
            x1_vol_pred_err = x1_vol_pred_err,
            x1_val_pred_err = x1_val_pred_err,
            x1_post_mean = x1_post_mean,
            x1_post_precis = x1_post_precis,

            x2_pred_precis = x2_pred_precis,
            x2_pred_vol = x2_pred_vol,
            x2_val_pred_err = x2_val_pred_err,
            x2_post_mean = x2_post_mean,
            x2_post_precis = x2_post_precis,

            x3_pred_precis = x3_pred_precis,
            x3_pred_vol = x3_pred_vol,
            x3_vol_pred_err = x3_vol_pred_err,
            x3_val_pred_err = x3_val_pred_err,
            x3_post_mean = x3_post_mean,
            x3_post_precis = x3_post_precis,

            x4_pred_precis = x4_pred_precis,
            x4_pred_vol = x4_pred_vol,
            x4_val_pred_err = x4_val_pred_err,
            x4_post_mean = x4_post_mean,
            x4_post_precis = x4_post_precis
        )

        df_get_history_extracts = vcat(df_get_history_extracts, data_frame_get_history_extracts)
        amount_of_succes = amount_of_succes + 1
    end
end

big_joined_df = leftjoin(df, df_get_history_extracts, on = [:ID,:session,:outcome,:trials])


CSV.write("/Users/ottosejrskildsantesson/Desktop/Bachelor/HGF_Bachelor/data/big_joined_df.csv", big_joined_df)