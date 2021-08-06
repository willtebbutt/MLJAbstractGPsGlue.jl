module MLJAbstractGPsGlue

using AbstractGPs
using MLJModelInterface
using Optim
using ParameterHandling
using Zygote

import MLJModelInterface:
    fit,
    fitted_params,
    predict,
    predict_joint,
    predict_mean,
    matrix,
    JointProbabilistic,
    target_scitype,
    input_scitype

mutable struct MLJAbstractGP{T, V} <: JointProbabilistic
    initial_parameters::T
    build_gp::V
end

function MLJAbstractGP(;
    initial_gp_parameters=__default_gp_parameters,
    build_gp=__default_build_gp,
    initial_noise_variance::Real=1.0,
)
    if initial_noise_variance <= 0
        throw(error("noise variance must be positive."))
    end
    if !(build_gp(ParameterHandling.value(initial_gp_parameters)) isa AbstractGPs.AbstractGP)
        throw(error("build_gp doesn't appear to build an AbstractGP."))
    end

    return MLJAbstractGP(
        (
            gp_parameters=initial_gp_parameters,
            noise_variance=positive(initial_noise_variance),
        ),
        build_gp,
    )
end

MLJModelInterface.clean!(model::MLJAbstractGP) = ""

# The default choise is a straightforward GP with an Exponentiated Quadratic kernel.
const __default_gp_parameters = (
    λ = bounded(1.0, 1e-2, 1e2),
    σ² = bounded(1.0, 1e-2, 1e2),
)

__default_build_gp(θ) = GP(θ.σ² * SEKernel() ∘ ScaleTransform(θ.λ))

# The scitype for these models could pretty much be anything, depending upon the kind of
# GP the user provides, so not possible to know this in general.
input_scitype(::Type{<:MLJAbstractGP}) = Unknown

# No explicit constraint placed up X, just requires that the `matrix` function works on it.
function fit(model::MLJAbstractGP, verbosity, X, y::AbstractVector{<:Real})

    # Convert inputs into a type that `AbstractGPs` understands.
    x = ColVecs(collect(matrix(X; transpose=true)))

    # Ensure that the inputs and targets are the same length.
    if length(x) != length(y)
        throw(error("number of inputs not the same as number of outputs"))
    end

    # Get flat parameters and functionality to reconstruct the unflattened counterpart.
    flat_initial_params, unflatten = flatten(model.initial_parameters)

    # Specify functionality to unpack and strip out all ParameterHandling-related data.
    unpack = ParameterHandling.value ∘ unflatten

    # Specify objective function.
    function objective(params)
        f = model.build_gp(params.gp_parameters)
        return -logpdf(f(x, params.noise_variance + 1e-3), y)
    end

    # Use Optim.jl to optimise the hyperparameters. We're very opinionated and use BFGS.
    training_results = Optim.optimize(
        objective ∘ unpack,
        θ -> only(Zygote.gradient(objective ∘ unpack, θ)),
        flat_initial_params,
        BFGS(
            alphaguess = Optim.LineSearches.InitialStatic(scaled=true),
            linesearch = Optim.LineSearches.BackTracking(),
        ),
        Optim.Options(show_trace = verbosity > 0);
        inplace=false,
    )

    # Extract parameters and construct the posterior at the optimum.
    final_params = unpack(training_results.minimizer)
    fx = model.build_gp(final_params.gp_parameters)(x, final_params.noise_variance + 1e-3)
    f_post = posterior(fx, y)

    # Compile thing to return and return them.
    fit_result = (
        posterior=f_post,
        noise_variance=final_params.noise_variance,
        final_params=final_params,
    )
    cache = nothing
    report = (training_results=training_results, final_params=final_params)
    return fit_result, cache, report
end

fitted_params(::MLJAbstractGP, fit_result) = fit_result.final_params

function predict_joint(model::MLJAbstractGP, fit_result, X_new)

    # Convert inputs into a type that `AbstractGPs` understands.
    x_new = ColVecs(collect(matrix(X_new; transpose=true)))

    # Produce a posterior prediction. This is a FiniteGP, which is a multivariate Normal.
    return fit_result.posterior(x_new, fit_result.noise_variance + 1e-3)
end

function predict(model::MLJAbstractGP, fit_result, X_new)
    return marginals(predict_joint(model, fit_result, X_new))
end

function predict_mean(model::MLJAbstractGP, fit_result, X_new)
    return map(mean, predict(model, fit_result, X_new))
end

logpdf_loss(marginals, y) = -sum((d, y) -> logpdf(d, y), zip(marginals, y))

MLJModelInterface.metadata_pkg(
    MLJAbstractGP;
    name="AbstractGPs.jl",
    uuid="99985d1d-32ba-4be9-9821-2ec096f28918",
    url="https://github.com/willtebbutt/AbstractGPs.jl/",
    julia=true,
    license="MIT",
    is_wrapper=true,
)

MLJModelInterface.metadata_model(
    MLJAbstractGP;
    target_scitype=AbstractVector{MLJModelInterface.Continuous},
    supports_weights=false,
    descr="Vanilla GP regression using any model in the AbstractGPs interface.",
    load_path="MLJAbstractGPsGlue.MLJAbstractGP",
)


export MLJAbstractGP, logpdf_loss

end
