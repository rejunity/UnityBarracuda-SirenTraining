using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class Runner : MonoBehaviour
{
    public NNModel model;
    Model m_model;
    IWorker m_worker;
    public static WorkerFactory.Type ms_workerType = WorkerFactory.Type.Compute;

    public bool loadModelFromOnnx = false;
    private bool biasOutputAndTarget = false;
    public bool learnBias = true; // disabling learnable bias makes model run twice faster with Barracuda 1.3.2 and older.

    public float learningRate = 0.001f; // Good value for Adam use 0.001, for SGD use 0.05f
    public float beta1 = 0.9f; // Adam only
    public float beta2 = 0.999f; // Adam only
    public float epsilon = 10e-8f; // Adam only
    private int totalSteps = 150;

    public Texture targetImage;
    public RenderTexture resultRT;

    Tensor m_input;
    Tensor m_target;
    Tensor m_lr;
    Tensor m_beta1;
    Tensor m_beta2;
    Tensor m_epsilon;
    public bool useAdam = false;
    Dictionary<string, Tensor> m_parameters;
    float m_lastUpdateTime;

    void InitGrid(ref Tensor X, int h, int w)
    {
        var end = X.length;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                X[2 * (y * w + x) + 0] = 2.0f * ((float)y) / ((float)h) - 1.0f;
                X[2 * (y * w + x) + 1] = 2.0f * ((float)x) / ((float)w) - 1.0f;
            }
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
        if (optimizerText)
            optimizerText.SetText(useAdam ? "ADAM" : "SGD");

        var batch = targetImage.height * targetImage.width;
        var sirenBuilder = new SirenModel(batch, biasOutputAndTarget, useAdam:useAdam, trainableBias:learnBias);
        m_model = sirenBuilder.model;
        m_parameters = sirenBuilder.parameters;

        if (loadModelFromOnnx)
            m_model = ModelLoader.Load(model, false);
        m_worker = WorkerFactory.CreateWorker(ms_workerType, m_model, false);

        m_input = new Tensor(batch, 2);
        InitGrid(ref m_input, targetImage.height, targetImage.width);

        m_target = new Tensor(targetImage, channels: 1);
        m_target = m_target.Reshape(new TensorShape(batch, 1));
        m_lr = new Tensor(1, 1, new[] { learningRate });
        m_beta1 = new Tensor(1, 1, new[] { beta1 });
        m_beta2 = new Tensor(1, 1, new[] { beta2 });
        m_epsilon = new Tensor(1, 1, new[] { epsilon });

        m_worker.SetInput("input", m_input);
        m_worker.SetInput("target", m_target);
        m_worker.SetInput("lr", m_lr);
        m_worker.SetInput("beta1", m_beta1);
        m_worker.SetInput("beta2", m_beta2);
        m_worker.SetInput("epsilon", m_epsilon);

        InitPlot();

        m_lastUpdateTime = Time.realtimeSinceStartup;
    }

    private int trainingSteps = 0;
    void Update()
    {
        if (trainingSteps >= totalSteps)
            return;

        //Debug.Log($"step: {trainingSteps} learning rate: {learningRate} step time: {Time.realtimeSinceStartup - m_lastUpdateTime}");
        m_lastUpdateTime = Time.realtimeSinceStartup;

        m_worker.Execute(m_parameters);

        UpdateParameters();

        var output = m_worker.PeekOutput("output");
        output = output.Reshape(new TensorShape(1, targetImage.height, targetImage.width, 1));
        output.ToRenderTexture(resultRT, batch: 0, fromChannel:0);

        PlotLoss(trainingSteps, m_worker.PeekOutput("loss")[0]);

        trainingSteps++;
    }

    void OnDestroy()
    {
        m_input.Dispose();
        m_target.Dispose();

        m_lr.Dispose();
        m_beta1.Dispose();
        m_beta2.Dispose();
        m_epsilon.Dispose();

        foreach (var param in m_parameters)
            param.Value.Dispose();

        m_worker.Dispose();

        DisposePlot();
    }

    // Plot

    ComputeBuffer lossBufferGPU;
    float[] lossBufferCPU;
    float maxValueLoss = 0.0f;
    public ComputeShader graphPlotter;
    RenderTexture graphRT;
    public Material graphMaterial;
    public Color lossColor = Color.red;
    public TMPro.TextMeshPro lossText;
    public TMPro.TextMeshPro iterationText;
    public TMPro.TextMeshPro optimizerText;


    void InitPlot()
    {
        graphRT = new RenderTexture(1024, 1024, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        graphRT.enableRandomWrite = true;
        graphRT.Create();

        lossBufferGPU = new ComputeBuffer(totalSteps, sizeof(float));
        lossBufferCPU = new float[totalSteps];
    }

    void PlotLoss(int step, float loss)
    {
        if (iterationText)
            iterationText.SetText($"iteration:{step}");
        if (lossText)
            lossText.SetText($"loss:{loss}");

        lossBufferCPU[step] = loss;
        maxValueLoss = Mathf.Max(maxValueLoss, loss);
        lossBufferGPU.SetData(lossBufferCPU, 0, 0, totalSteps);

        int kernelHandle = graphPlotter.FindKernel("CSMain");

        graphPlotter.SetTexture(kernelHandle, "graphTexture", graphRT);
        graphPlotter.SetBuffer(kernelHandle, "graphBuffer", lossBufferGPU);
        graphPlotter.SetInt("graphDimX", graphRT.width);
        graphPlotter.SetInt("graphDimY", graphRT.height);
        graphPlotter.SetInt("graphBufferTotalCount", totalSteps);
        graphPlotter.SetInt("graphBufferValueCount", step + 1);
        graphPlotter.SetFloat("maxValue", maxValueLoss);
        graphPlotter.SetFloats("backgroundColor", new float[]{Camera.main.backgroundColor.r, Camera.main.backgroundColor.g, Camera.main.backgroundColor.b, Camera.main.backgroundColor.a});
        graphPlotter.SetFloats("lossColor", new float[]{lossColor.r, lossColor.g, lossColor.b, lossColor.a});

        graphPlotter.Dispatch(kernelHandle, graphRT.width / 8, graphRT.height / 8, 1);
        graphMaterial.mainTexture = graphRT;
    }

    void DisposePlot()
    {
        lossBufferGPU.Dispose();
        graphRT.Release();
    }

}
