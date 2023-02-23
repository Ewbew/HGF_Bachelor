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

fixed_params = Dict(
    ("u", "x1", "value_coupling")       => 1, 
    ("x4", "evolution_rate")            => -15, 
    ("x4", "initial_mean")              => 1,
    ("x3", "x4", "volatility_coupling") => 1,
    ("u", "evolution_rate")             => 0,
    ("x4", "initial_precision")         => 1/0.1,
    ("x1", "initial_precision")         => 1/3, 
    ("x2", "initial_precision")         => 1/0.1,
    ("x3", "initial_precision")         => 1/3, 
    ("x3", "initial_mean")              => log(25), 
    ("x2", "initial_mean")              => 3,
    ("x1", "x2", "volatility_coupling") => 1,
    ("u", "x3", "volatility_coupling")  => 1,
);


param_priors = Dict(
    ("x1", "evolution_rate")            => Normal(-1.65,0.5), 
    ("x3", "evolution_rate")            => Normal(-6.3,0.75),
    ("x2", "evolution_rate")            => Normal(-5,0.35),
    "gaussian_action_precision"         => Gamma(2,0.1), 
);


data_path_loop = "/Users/ottosejrskildsantesson/Desktop/Bachelor/HGF_Bachelor/data/fitted_models/"


# A for-loop that goes through every combination of participant ID and session number,
# making a fit for each combination, and saving the fit to a file
for ID in unique(df[!,:ID])
    for session in unique(df[!,:session])


        # filter the data for the current session and participant
        filtered_data = filter([:ID, :session] => (x, y) -> x == ID && y == session , df)

        # Skips the iteration, if the filtered data has no rows
        if size(filtered_data,1) == 0
            continue
        end

        fixed_params = Dict(
        ("u", "x1", "value_coupling")       => 1, 
        ("x4", "evolution_rate")            => -15, 
        ("x4", "initial_mean")              => 1,
        ("x3", "x4", "volatility_coupling") => 1,
        ("u", "evolution_rate")             => 0,
        ("x4", "initial_precision")         => 1/0.1,
        ("x1", "initial_precision")         => 1/3, 
        ("x2", "initial_precision")         => 1/0.1,
        ("x1", "initial_mean")              => filtered_data[1,:outcome], # The initial mean is set to the first input
        ("x3", "initial_precision")         => 1/3, 
        ("x3", "initial_mean")              => log(25),
        ("x2", "initial_mean")              => 3,
        ("x1", "x2", "volatility_coupling") => 1,
        ("u", "x3", "volatility_coupling")  => 1,
        );

        # Taking the inputs of the data (u in the HGF, what the participant observes)
        inputs = filtered_data[!,:outcome]

        # Choosing the response (actions) of the data
        actions = filtered_data[!,:response]

        # Fitting the model to the filtered data for the unique ID & session combination
        fitted_model = fit_model(agent,inputs,actions,param_priors,fixed_params, n_iterations = 1000,verbose = true, n_chains = 4)

        # Defining the filename to be dependent on the participant ID and session number
        filename = "participant_$(ID)_session_$(session).jld2"

        # Saving the file in the folder "data/fitted_models"
        save_object(data_path_loop * filename, fitted_model)
    end
end
