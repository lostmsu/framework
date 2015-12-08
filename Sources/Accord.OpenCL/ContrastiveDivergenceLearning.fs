namespace Accord.OpenCL

open FSCL
open FSCL.Compiler
open FSCL.Language
open FSCL.Runtime

module Kernels =
    [<ReflectedDefinition; Kernel>]
    let ComputePositiveAssociations(observation: float32[], probability: float32[], weightGradient: float32[,],
            wi: WorkItemInfo) =

        let k = wi.GlobalID(0)
        let newGradient = Array2D.zeroCreate observation.Length probability.Length

        for j = 0 to probability.Length - 1 do
            newGradient.[k, j] <- weightGradient.[k, j] + observation.[k]*probability.[j]

        newGradient

type ContrastiveDivergenceLearning() =
    member this.ComputePositiveAssociations(observation: double[], probability: double[],
            weightGradient: double[][], hiddenGradient: double[], visibleGradient: double[]) =
        let observation32 = Array.map float32 observation
        let probability32 = Array.map float32 probability
        let weightGradient32Old = Array2D.init weightGradient.Length weightGradient.[0].Length 
                                            (fun i j -> float32 weightGradient.[i].[j])
        let workSize = WorkSize(observation32.LongLength)
        
        let weightGradient32 = <@ Kernels.ComputePositiveAssociations(observation32, probability32, weightGradient32Old, workSize) @>.Run()

        let weightGradient32' = Array2D.init weightGradient.Length weightGradient.[0].Length 
                                            (fun i j -> float32 weightGradient.[i].[j])
        let mutable totalDiff = 0.0f
        Array2D.iteri (fun i j newGrad -> totalDiff <- totalDiff + abs(newGrad - weightGradient32'.[i, j])) weightGradient32
        System.Diagnostics.Trace.WriteLine(totalDiff)

        Array2D.iteri (fun i j newGrad -> weightGradient.[i].[j] <- double newGrad) weightGradient32

        for j = 0 to hiddenGradient.Length - 1 do
            hiddenGradient.[j] <- hiddenGradient.[j] + probability.[j]

        for j = 0 to visibleGradient.Length - 1 do
            visibleGradient.[j] <- visibleGradient.[j] + observation.[j]