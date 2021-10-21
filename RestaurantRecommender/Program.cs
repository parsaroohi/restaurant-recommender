using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace RestaurantRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Restaurants recommender");

            var mlContext = new MLContext(0);

            var trainingDataFile = "Data\\trainingData.tsv";

            DataPreparer.PreprocessData(trainingDataFile);

            IDataView trainingDataView = mlContext.Data
                .LoadFromTextFile<ModelInput>(
                    trainingDataFile,
                    hasHeader: true
                );

            var dataPreProcessingPipeline = mlContext
                .Transforms.Conversion
                .MapValueToKey(outputColumnName: "UserIdEncoded", inputColumnName: nameof(ModelInput.UserId))
                .Append(mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "RestaurantNameEncoded", inputColumnName: nameof(ModelInput.RestaurantName)));

            var finalOption = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "UserIdEncoded",
                MatrixRowIndexColumnName = "RestaurantNameEncoded",
                LabelColumnName = "TotalRating",
                NumberOfIterations = 10,
                ApproximationRank = 200,
                Quiet = true
            };

            var trainer = mlContext.Recommendation()
                .Trainers.MatrixFactorization(finalOption);

            var trainerPipeline = dataPreProcessingPipeline.Append(trainer);
            Console.WriteLine("Training Model");

            var model = trainerPipeline.Fit(trainingDataView);

            ////Test Result
            //var testUserId = "U1134";
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<ModelInput, ModelOutput>(model);

            //var alreadyRatedRestaurant = mlContext.Data
            //    .CreateEnumerable<ModelInput>(trainingDataView, false)
            //    .Where(i => i.UserId == testUserId)
            //    .Select(r => r.RestaurantName)
            //    .Distinct();

            //var allRestaurantNames =
            //    trainingDataView.GetColumn<string>("RestaurantName")
            //    .Distinct()
            //    .Where(r => !alreadyRatedRestaurant.Contains(r));

            //var scoredRestaurants =
            //    allRestaurantNames.Select(restName =>
            //    {
            //        var prediction = predictionEngine.Predict(
            //            new ModelInput()
            //            {
            //                RestaurantName = restName,
            //                UserId = testUserId,
            //            });
            //        return (RestaurantName: restName, PredictedRating: prediction.Score);
            //    });

            //var top10Restaurants = scoredRestaurants
            //    .OrderByDescending(s => s.PredictedRating)
            //    .Take(10);

            //Console.WriteLine();
            //Console.WriteLine($"Top 10 restaurants for {testUserId}");
            //Console.WriteLine("-------------------------------");
            //foreach (var top in top10Restaurants)
            //{
            //    Console.WriteLine($"Predicted Rating [{top.PredictedRating:#.0}] for restaurant {top.RestaurantName}");
            //}

            var crossValMetrics = mlContext.Recommendation()
                .CrossValidate(data: trainingDataView,
                estimator: trainerPipeline,
                labelColumnName: "TotalRating");

            var averageRMSE = crossValMetrics.Average(m => m.Metrics.RootMeanSquaredError);
            var averageRSquared = crossValMetrics.Average(m => m.Metrics.RSquared);

            Console.WriteLine();
            Console.WriteLine("--- Metrics before tuning hyper parameters ---");
            Console.WriteLine($"Cross validated root error : {averageRMSE:#.000}");
            Console.WriteLine($"Cross validated RSquerd : {averageRSquared:#.000}");
            Console.WriteLine();

            //HyperParameterExploration(mlContext, dataPreProcessingPipeline, trainingDataView);

            var prediction = predictionEngine
                .Predict(new ModelInput()
                {
                    UserId = "CLONED",
                    RestaurantName = "Restaurant Wu Zhuo Yi"
                });

            Console.WriteLine($"Predicted {prediction.Score:#.0} for Restaurant Wu Zhuo Yi");

        }


        private static void HyperParameterExploration(MLContext mlContext,
            IEstimator<ITransformer> dataProcessingPipeline,
            IDataView trainingDataView)
        {
            var results = new List<(double rootMeanSquaredError,
                double rSquerd,
                int iterations,
                int approximationRank)>();

            for (int iterations = 5; iterations < 100; iterations += 5)
            {
                for (int approximationRank = 50; approximationRank < 250; approximationRank += 50)
                {
                    var option = new MatrixFactorizationTrainer.Options
                    {
                        MatrixColumnIndexColumnName = "UserIdEncoded",
                        MatrixRowIndexColumnName = "RestaurantNameEncoded",
                        LabelColumnName = "TotalRating",
                        NumberOfIterations = iterations,
                        ApproximationRank = approximationRank,
                        Quiet = true
                    };

                    var currentTrainer = mlContext
                        .Recommendation().Trainers.MatrixFactorization(option);

                    var completePipeline = dataProcessingPipeline.Append(currentTrainer);

                    var crossValMetrics = mlContext.Recommendation()
                        .CrossValidate(trainingDataView, completePipeline, labelColumnName: "TotalRating");

                    results.Add((
                        crossValMetrics.Average(m => m.Metrics.RootMeanSquaredError),
                        crossValMetrics.Average(m => m.Metrics.RSquared),
                        iterations,
                        approximationRank
                    ));
                }
            }

            Console.WriteLine("--- Hyper Parameter and Metrics");

            foreach (var result in results.OrderByDescending(r => r.rSquerd))
            {
                Console.WriteLine($"NumberOfIterations : {result.iterations}");
                Console.WriteLine($"ApprximationRank : {result.approximationRank}");
                Console.WriteLine($"RootMeanSquerdError : {result.rootMeanSquaredError}");
                Console.WriteLine($"RSquerd : {result.rSquerd}");
            }

            Console.WriteLine();
            Console.WriteLine("Done!");

        }
    }
}
