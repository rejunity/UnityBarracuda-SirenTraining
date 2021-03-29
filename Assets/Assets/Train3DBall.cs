using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class Train3DBall : MonoBehaviour
{
    public NNModel model;
    Model m_model;
    IWorker m_worker;
    public static WorkerFactory.Type ms_workerType = WorkerFactory.Type.ComputePrecompiled;

    public bool loadModelFromOnnx = false;
    public bool biasOutputAndTarget = true;

    public float learningRate = 0.01f;
    private int totalSteps = 150;

    public RenderTexture targetRT;
    public RenderTexture resultRT;

    Tensor m_input;
    Tensor m_target;
    Tensor m_lr;
    Dictionary<string, Tensor> m_parameters;
    float m_lastUpdateTime;
    private Tuple<List<Tensor>, List<Tensor>> m_DataSet;
    SirenModel m_SirenBuilder;

    Tuple<List<Tensor>, List<Tensor>> LoadDataSet()
    {
        var res_input = new List<Tensor>();
        var res_output = new List<Tensor>();

        for (var idx = 0; idx < 10; idx++)
        {
            var testset = TestSetLoader.Load($"TrainingData/3DBallSupervised/input{idx}.json");
            var ti = testset.GetInputAsTensor();
            var to = testset.GetOutputAsTensor();
            res_input.Add(ti);
            res_output.Add(to);
        }

        return new Tuple<List<Tensor>, List<Tensor>>(res_input, res_output);
    }

    void SlowCopy(Tensor X, ref Tensor O)
    {
        var end = X.length;
        for (int i = 0; i < end; ++i)
            O[i] = X[i];
    }

    void UpdateParameters(string prefixForUpdatedParameter = "new_")
    {
        foreach (var param in m_parameters)
        {
            var value = param.Value;
            var newParam = $"{prefixForUpdatedParameter}{param.Key}";
            SlowCopy(m_worker.PeekOutput(newParam), ref value);
        }
    }

    void Start()
    {
        m_DataSet = LoadDataSet();

        var batch = m_DataSet.Item1[0].batch;
        m_SirenBuilder = new SirenModel(batch, false, 256, 3,
            8, 2, 6,
            "vector_observation", "continuous_actions", false);
        m_model = m_SirenBuilder.model;
        m_parameters = m_SirenBuilder.parameters;

        if (loadModelFromOnnx)
            m_model = ModelLoader.Load(model, false);
        m_worker = WorkerFactory.CreateWorker(ms_workerType, m_model, false);

        m_input = m_DataSet.Item1[9];
        m_target = m_DataSet.Item2[9];

        m_lr = new Tensor(1, 1, new[] { learningRate });

        InitPlot();

        var t = m_target.Reshape(new TensorShape(1, 100, 200, 1));
        t.ToRenderTexture(targetRT, batch: 0, fromChannel: 0);

        m_lastUpdateTime = Time.realtimeSinceStartup;

        StartCoroutine(TrainingLoop());
    }

    private int trainingSteps = 0;

    IEnumerator TrainingLoop()
    {
        while (true)
        {
            if (trainingSteps >= totalSteps)
                yield break;

            for (var idx = 0; idx < m_DataSet.Item1.Count; idx++)
            {
                Debug.Log(
                    $"step: {trainingSteps} learning rate: {learningRate} step time: {Time.realtimeSinceStartup - m_lastUpdateTime}");
                m_lastUpdateTime = Time.realtimeSinceStartup;

                m_input = m_DataSet.Item1[idx];
                m_target = m_DataSet.Item2[idx];

                m_worker.SetInput("vector_observation", m_input);
                m_worker.SetInput("target", m_target);
                m_worker.SetInput("lr", m_lr);
                m_worker.Execute(m_parameters);

                UpdateParameters();

                PlotLoss(trainingSteps, m_worker.PeekOutput("loss")[0]);
                yield return null;
            }

            var output = m_worker.PeekOutput("continuous_actions");
            output = output.Reshape(new TensorShape(1, 100, 200, 1));
            output.ToRenderTexture(resultRT, batch: 0, fromChannel: 0);



            trainingSteps++;
        }
    }

    void OnDestroy()
    {
        ModelWriter.Save("Assets/3DBall.nn", m_SirenBuilder.BuildMLAgentsModel(256, 3, 8, 2, 6));
        m_input.Dispose();
        m_target.Dispose(); m_lr.Dispose();

        foreach (var param in m_parameters)
            param.Value.Dispose();

        m_worker.Dispose();

        DisposePlot();
    }

    // Plot

    ComputeBuffer lossBufferGPU;
    float[] lossBufferCPU;
    public ComputeShader graphPlotter;
    RenderTexture graphRT;
    public Material graphMaterial;

    void InitPlot()
    {
        graphRT = new RenderTexture(256, 32, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        graphRT.enableRandomWrite = true;
        graphRT.Create();

        lossBufferGPU = new ComputeBuffer(totalSteps, sizeof(float));
        lossBufferCPU = new float[totalSteps];
    }

    void PlotLoss(int step, float loss)
    {
        Debug.Log($"Loss: {loss}");
        lossBufferCPU[step] = loss;
        lossBufferGPU.SetData(lossBufferCPU, 0, 0, totalSteps);

        int kernelHandle = graphPlotter.FindKernel("CSMain");

        graphPlotter.SetTexture(kernelHandle, "graphTexture", graphRT);
        graphPlotter.SetBuffer(kernelHandle, "graphBuffer", lossBufferGPU);
        graphPlotter.SetInt("graphDimX", graphRT.width);
        graphPlotter.SetInt("graphDimY", graphRT.height);
        graphPlotter.SetInt("graphBufferCount", totalSteps);

        graphPlotter.Dispatch(kernelHandle, graphRT.width / 8, graphRT.height / 8, 1);
        graphMaterial.mainTexture = graphRT;
    }

    void DisposePlot()
    {
        lossBufferGPU.Dispose();
        graphRT.Release();
    }

}
