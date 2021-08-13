using AbstractGPs
using MLJAbstractGPsGlue
using MLJBase
using ParameterHandling
using Test

@testset "MLJAbstractGPsGlue.jl" begin

    X, y = @load_boston

    model = MLJAbstractGP()
    regressor = machine(model, X, y)

    # Just ensure that this runs as intended.
    evaluate!(regressor, resampling=CV(), measure=logpdf_loss, verbosity=0)

    # Train a single model and make some predictions.
    train_rows, test_rows = partition(eachindex(y), 0.7, shuffle=true)
    fit!(regressor; rows=train_rows, verbosity=0)

    @test predict_joint(regressor; rows=test_rows) isa AbstractGPs.AbstractMvNormal
    @test predict(regressor; rows=test_rows) isa Vector{<:AbstractGPs.Normal}
    @test predict_mean(regressor; rows=test_rows) isa Vector{<:Real}

    @testset for j in 1:10
        @test isapprox(
            mean(predict(regressor; rows=test_rows[1:10])[j]),
            mean(predict(regressor; rows=test_rows)[j]),
        )
        @test isapprox(
            std(predict(regressor; rows=test_rows[1:10])[j]),
            std(predict(regressor; rows=test_rows)[j]),
        )
    end
end
