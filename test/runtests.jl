using AbstractGPs
using MLJAbstractGPsGlue
using MLJBase
using MLJ
using ParameterHandling
using Test

@testset "MLJAbstractGPsGlue.jl" begin

    X, y = @load_boston

    model = MLJAbstractGP()
    regressor = machine(model, X, y)

    # Just ensure that this runs as intended.
    # evaluate!(regressor, resampling=CV(), measure=logpdf_loss, verbosity=0)

    # Train a single model and make some predictions.
    train_rows, test_rows = partition(eachindex(y), 0.7, shuffle=true)
    fit!(regressor; rows=train_rows)
    display(predict(regressor, X[test_rows, :]))
    println()
end
