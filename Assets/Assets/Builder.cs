using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

public class SirenModel
{
    public Model model;
    public List<Tensor> weight      = new List<Tensor>();
    public List<Tensor> bias        = new List<Tensor>();
    public List<Tensor> moment_w    = new List<Tensor>();
    public List<Tensor> moment_b    = new List<Tensor>();
    public List<Tensor> velocity_w  = new List<Tensor>();
    public List<Tensor> velocity_b  = new List<Tensor>();
    public Tensor beta1t;
    public Tensor beta2t;
    public Dictionary<string, Tensor> parameters = new Dictionary<string, Tensor>();

    void InitZero(ref Tensor X)
    {
        var end = X.length;
        for (int i = 0; i < end; ++i)
            X[i] = 0;
    }

    void InitUniform(ref Tensor X, float low, float high)
    {
        var end = X.length;
        for (int i = 0; i < end; ++i)
            X[i] = (high - low) * UnityEngine.Random.value + low;
    }

    void AddToTrainableParametersAndAssignIndexedName(ModelBuilder net, List<Tensor> tensors, string prefix)
    {
        int index = 0;
        foreach (var t in tensors)
        {
            t.name = $"{prefix}{index++}";
            net.Input(t.name, t.shape);
            parameters.Add(t.name, t);
        }
    }

    void AddToTrainableParametersAndAssignIndexedName(ModelBuilder net, Tensor tensor, string prefix)
    {
        tensor.name = prefix;
        net.Input(tensor.name, tensor.shape);
        parameters.Add(tensor.name, tensor);
    }

    List<Tensor> CreateMoments(List<Tensor> tensors)
    {
        List<Tensor> moments = new List<Tensor>();
        // create zero initialized tensors for moments
        foreach (var t in tensors)
            moments.Add(new Tensor(t.shape, new float[t.length]));
        return moments;
    }

    public SirenModel(
        int batch,
        bool biasOutputAndTarget = false,
        int hidden_features = 256,
        int hidden_layers = 3,
        int in_features = 2,
        int out_features = 1,
        float omega_0 = 30.0f,
        string input_name = "input",
        string output_name = "output",
        bool useAdam = true,
        bool trainableBias = true)
    {
        // model tensors
        weight.Add( new Tensor(hidden_features, in_features));
        bias.Add(   new Tensor(1, hidden_features));
        for (int i = 0; i < hidden_layers; ++i)
        {
            weight.Add( new Tensor(hidden_features, hidden_features));
            bias.Add(   new Tensor(1, hidden_features));
        }
        weight.Add( new Tensor(out_features, hidden_features));
        bias.Add(   new Tensor(1, out_features));
        var layerCount = weight.Count;

        // initialize tensors
        {
            var w = weight[0];
            var b = bias[0];
            InitUniform(ref w, -1.0f / in_features, 1.0f / in_features);
            InitUniform(ref b, -1.0f / in_features, 1.0f / in_features);
        }
        for (int i = 1; i < layerCount; ++i)
        {
            var w = weight[i]; var b = bias[i];
            InitUniform(ref w, -Mathf.Sqrt(6.0f / hidden_features) / omega_0, Mathf.Sqrt(6.0f / hidden_features) / omega_0);
            InitUniform(ref b, -Mathf.Sqrt(6.0f / hidden_features) / omega_0, Mathf.Sqrt(6.0f / hidden_features) / omega_0);
        }

        // build model
        var net = new ModelBuilder();
        var ctx = new Stack<object>();

        // setup tensors as trainable parameters, constants & inputs
        AddToTrainableParametersAndAssignIndexedName(net, weight, "w");
        AddToTrainableParametersAndAssignIndexedName(net, bias, "b");
        if (useAdam)
        {
            beta1t = new Tensor(new TensorShape(1, 1), new float[1] { 1 });
            beta2t = new Tensor(new TensorShape(1, 1), new float[1] { 1 });
            moment_w = CreateMoments(weight);
            moment_b = CreateMoments(bias);
            velocity_w = CreateMoments(weight);
            velocity_b = CreateMoments(bias);
            AddToTrainableParametersAndAssignIndexedName(net, beta1t, "beta1t");
            AddToTrainableParametersAndAssignIndexedName(net, beta2t, "beta2t");
            AddToTrainableParametersAndAssignIndexedName(net, moment_w, "moment_w");
            AddToTrainableParametersAndAssignIndexedName(net, velocity_w, "velocity_w");

            if (trainableBias)
            {
                AddToTrainableParametersAndAssignIndexedName(net, moment_b, "moment_b");
                AddToTrainableParametersAndAssignIndexedName(net, velocity_b, "velocity_b");
            }
        }

        var Omega0 = net.Const("omega_0", new Tensor(1, 1, new float[] { omega_0 }));
        var One = net.Const("one", new Tensor(1, 1, new float[] { 1f }));
        var Double = net.Const("two", new Tensor(1, 1, new float[] { 2f }));
        var Half = net.Const("half", new Tensor(1, 1, new float[] { 0.5f }));
        var InvBatchDouble = net.Const("invBatchDouble", new Tensor(1, 1, new float[] { 2.0f / batch }));
        var Batch = net.Const("batch", new Tensor(1, 1, new float[] { batch }));
        var N__C2C__N = new[] {3,1,2,0}; // transpose weights by swapping N and C channels

        object lr = net.Input("lr", 1, 1);
        object beta1 = net.Input("beta1", 1, 1);
        object beta2 = net.Input("beta2", 1, 1);
        object epsilon = net.Input("epsilon", 1, 1);
        object x = net.Input(input_name, batch, in_features);

        // forward
        for (int i = 0; i < layerCount; ++i)
        {
            var isLastLayer = i == layerCount - 1;
            var w = weight[i].name; var b = bias[i].name;

            ctx.Push(x);
            x = net.MatMul($"mm{i}", x, net.Transpose($"{w}.T", w, N__C2C__N));
            x = net.Add($"bias{i}", new[] {x, b});
            if (!isLastLayer)
            {
                ctx.Push(x);
                x = net.Mul($"sin_premul{i}_omega", new[] {x, Omega0});
                x = net.Sin($"sin{i}", x);
            }
        }
        object output = x;
        object target = net.Input("target", batch, out_features);

        if (biasOutputAndTarget)
        {
            output = net.Mul($"output_mul_0.5", new[] {output, Half});
            output = net.Add($"output_add_0.5", new[] {output, Half});

            target = net.Mul($"target_mul_2", new[] {target, Double});
            target = net.Sub($"target_sub_1", new[] {target, One});
        }
        net.Output(net.Identity(output_name, output));

        // loss
        var error = net.Sub("error", new[] {x, target});
        net.Output(net.Reduce(Layer.Type.ReduceMean, "loss",
                   net.Mul("error_sq", new[] {error, error}), axis:0));
        object grad_output = net.Mul("loss_grad", new[] {error, InvBatchDouble});

        // backward
        if (useAdam)
        {
            object b1t = beta1t.name;
            object beta1tp1 = net.Mul($"new_{b1t}", new[] { beta1, b1t });
            net.Output(beta1tp1);
            //object nbeta1tp1 = net.Sub("nbeta1tp1", new[] { One, beta1tp1 });

            object b2t = beta2t.name;
            object beta2tp1 = net.Mul($"new_{b2t}", new[] { beta2, b2t });
            net.Output(beta2tp1);
            //object nbeta2tp1 = net.Sub("nbeta2tp1", new[] { One, beta2tp1 });

            object nbeta1t = net.Sub("nbeta1t", new[] { One, beta1tp1 });
            object nbeta2t = net.Sub("nbeta2t", new[] { One, beta2tp1 });
            object nbeta1 = net.Sub("nbeta1", new[] { One, beta1 });
            object nbeta2 = net.Sub("nbeta2", new[] { One, beta2 });

            for (int i = layerCount - 1; i >= 0; --i)
            {
                var isLastLayer = i == layerCount - 1;
                var w = weight[i].name; var b = bias[i].name;

                if (!isLastLayer)
                {
                    var input = ctx.Pop();
                    input = net.Mul($"sin_grad_premul{i}_omega", new[] { input, Omega0 });
                    input = net.Cos($"sin_grad_cos{i}", input);
                    grad_output =
                       net.Mul($"sin_grad{i}", new[] { grad_output, input, Omega0 });
                }

                // weights
                object grad_w = net.MatMul($"grad_{w}",
                    net.Transpose($"grad_output{i}.T", grad_output, N__C2C__N), ctx.Pop());
                object grad_w2 = net.Mul($"grad_{w}2", new[] { grad_w, grad_w });

                object mom_w = moment_w[i].name;
                mom_w = net.Mul($"m_moment0_{w}", new[] { beta1, mom_w });
                grad_w = net.Mul($"m_grad0_{grad_w}", new[] { nbeta1, grad_w });
                mom_w = net.Add($"new_moment_{w}", new[] { mom_w, grad_w });
                net.Output(mom_w);

                mom_w = net.Div($"m_moment1_{w}", new[] { mom_w, nbeta1t });

                object vel_w = velocity_w[i].name;
                vel_w = net.Mul($"m_velocity0_{w}", new[] { beta2, vel_w });
                grad_w2 = net.Mul($"m_grad0_{grad_w2}", new[] { nbeta2, grad_w2 });
                vel_w = net.Add($"new_velocity_{w}", new[] { vel_w, grad_w2 });
                net.Output(vel_w);

                vel_w = net.Div($"m_velocity1_{w}", new[] { vel_w, nbeta2t });
                vel_w = net.Sqrt($"m_velocity2_{w}", vel_w);
                vel_w = net.Add($"m_velocity3_{w}", new[] { vel_w, epsilon });

                object etaw = net.Div($"etaw0_{w}", new[] { mom_w, vel_w });
                etaw = net.Mul($"etaw1_{w}", new[] { lr, etaw });
                net.Output(net.Sub($"new_{w}", new[] { w, etaw }));


                if (trainableBias)
                {
                    // bias
                    object grad_b = net.Reduce(Layer.Type.ReduceSum, $"grad_{b}", grad_output, axis: 0);
                    object grad_b2 = net.Mul($"grad_{b}2", new[] { grad_b, grad_b });


                    object mom_b = moment_b[i].name;
                    mom_b  = net.Mul($"m_moment0_{b}", new[] { beta1, mom_b });
                    grad_b = net.Mul($"m_grad0_{grad_b}", new[] { nbeta1, grad_b });
                    mom_b = net.Add($"new_moment_{b}", new[] { mom_b, grad_b });
                    net.Output(mom_b);

                    mom_b = net.Div($"m_moment1_{b}", new[] { mom_b, nbeta1t });

                    object vel_b = velocity_b[i].name;
                    vel_b = net.Mul($"m_velocity0_{b}", new[] { beta2, vel_b });
                    grad_b2 = net.Mul($"m_grad0_{grad_b2}", new[] { nbeta2, grad_b2 });
                    vel_b = net.Add($"new_velocity_{b}", new[] { vel_b, grad_b2 });
                    net.Output(vel_b);

                    vel_b = net.Div($"m_velocity1_{b}", new[] { vel_b, nbeta2t });
                    vel_b = net.Sqrt($"m_velocity2_{b}", vel_b);
                    vel_b = net.Add($"m_velocity3_{b}", new[] { vel_b, epsilon });

                    object etab = net.Div($"etab0_{b}", new[] { mom_b, vel_b });
                    etab = net.Mul($"etab1_{b}", new[] { lr, etab });

                    net.Output(net.Sub($"new_{b}", new[] { b, etab }));
                }
                else
                    net.Output(net.Identity($"new_{b}", b));

                if (i > 0)
                    grad_output = net.MatMul($"grad_output{i - 1}", grad_output, w);
            }
        }
        else
        {
            for (int i = layerCount - 1; i >= 0; --i)
            {
                var isLastLayer = i == layerCount - 1;
                var w = weight[i].name; var b = bias[i].name;

                if (!isLastLayer)
                {
                    var input = ctx.Pop();
                    input = net.Mul($"sin_grad_premul{i}_omega", new[] { input, Omega0 });
                    input = net.Cos($"sin_grad_cos{i}", input);
                    grad_output =
                       net.Mul($"sin_grad{i}", new[] { grad_output, input, Omega0 });
                }
                
                object grad_w = net.MatMul($"grad_{w}",
                                net.Transpose($"grad_output{i}.T", grad_output, N__C2C__N), ctx.Pop());
                grad_w = net.Mul($"lr_grad_{w}", new[] { lr, grad_w });
                net.Output(net.Sub($"new_{w}", new[] { w, grad_w }));

                if (trainableBias)
                {
                    object grad_b = net.Reduce(Layer.Type.ReduceSum, $"grad_{b}", grad_output, axis: 0);
                    grad_b = net.Mul($"lr_grad_{b}", new[] { lr, grad_b });
                    net.Output(net.Sub($"new_{b}", new[] { b, grad_b }));
                }
                else
                    net.Output(net.Identity($"new_{b}", b));

                if (i > 0)
                    grad_output = net.MatMul($"grad_output{i - 1}", grad_output, w);
            }
        }

        model = net.model;
        Debug.Log(model);
    }

    public Model BuildMLAgentsModel
    (
        int hidden_features = 256,
        int hidden_layers = 3,
        int in_features = 2,
        int out_features = 1,
        float omega_0 = 30.0f,
        string input_name = "vector_observation",
        string output_name = "continuous_actions")
    {
        var layerCount = weight.Count;

        // build model
        var net = new ModelBuilder();
        var ctx = new Stack<object>();

        var Omega0 = net.Const("omega_0", new Tensor(1, 1, new float[] { omega_0 }));
        var N__C2C__N = new[] {3,1,2,0}; // transpose weights by swapping N and C channels

        object x = net.Input(input_name, 1, in_features);

        // forward
        for (int i = 0; i < layerCount; ++i)
        {
            var isLastLayer = i == layerCount - 1;
            var w_name = weight[i].name; var b_name = bias[i].name;
            var w = net.Const(w_name, parameters[w_name]);
            var b = net.Const(b_name, parameters[b_name]);

            ctx.Push(x);
            x = net.MatMul($"mm{i}", x, net.Transpose($"{w}.T", w, N__C2C__N));
            x = net.Add($"bias{i}", new[] {x, b});
            if (!isLastLayer)
            {
                ctx.Push(x);
                x = net.Mul($"sin_premul{i}_omega", new[] {x, Omega0});
                x = net.Sin($"sin{i}", x);
            }
        }
        object output = x;
        net.Output(net.Identity(output_name, output));
        net.Output(net.Const("version_number", new Tensor(1,1, new float[] {2.0f})));
        net.Output(net.Const("continuous_action_output_shape", new Tensor(1,1, new float[] {out_features})));
        net.Output(net.Const("memory_size", new Tensor(1,1, new float[] {0})));

        return net.model;
    }
}
