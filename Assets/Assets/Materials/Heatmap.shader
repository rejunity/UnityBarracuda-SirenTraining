Shader "Unlit/Heatmap"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            static const float3  turbo_srgb_floats[256] = { float3(0.18995, 0.07176, 0.23217), float3(0.19483, 0.08339, 0.26149), float3(0.19956, 0.09498, 0.29024), float3(0.20415, 0.10652, 0.31844), float3(0.20860, 0.11802, 0.34607), float3(0.21291, 0.12947, 0.37314), float3(0.21708, 0.14087, 0.39964), float3(0.22111, 0.15223, 0.42558), float3(0.22500, 0.16354, 0.45096), float3(0.22875, 0.17481, 0.47578), float3(0.23236, 0.18603, 0.50004), float3(0.23582, 0.19720, 0.52373), float3(0.23915, 0.20833, 0.54686), float3(0.24234, 0.21941, 0.56942), float3(0.24539, 0.23044, 0.59142), float3(0.24830, 0.24143, 0.61286), float3(0.25107, 0.25237, 0.63374), float3(0.25369, 0.26327, 0.65406), float3(0.25618, 0.27412, 0.67381), float3(0.25853, 0.28492, 0.69300), float3(0.26074, 0.29568, 0.71162), float3(0.26280, 0.30639, 0.72968), float3(0.26473, 0.31706, 0.74718), float3(0.26652, 0.32768, 0.76412), float3(0.26816, 0.33825, 0.78050), float3(0.26967, 0.34878, 0.79631), float3(0.27103, 0.35926, 0.81156), float3(0.27226, 0.36970, 0.82624), float3(0.27334, 0.38008, 0.84037), float3(0.27429, 0.39043, 0.85393), float3(0.27509, 0.40072, 0.86692), float3(0.27576, 0.41097, 0.87936), float3(0.27628, 0.42118, 0.89123), float3(0.27667, 0.43134, 0.90254), float3(0.27691, 0.44145, 0.91328), float3(0.27701, 0.45152, 0.92347), float3(0.27698, 0.46153, 0.93309), float3(0.27680, 0.47151, 0.94214), float3(0.27648, 0.48144, 0.95064), float3(0.27603, 0.49132, 0.95857), float3(0.27543, 0.50115, 0.96594), float3(0.27469, 0.51094, 0.97275), float3(0.27381, 0.52069, 0.97899), float3(0.27273, 0.53040, 0.98461), float3(0.27106, 0.54015, 0.98930), float3(0.26878, 0.54995, 0.99303), float3(0.26592, 0.55979, 0.99583), float3(0.26252, 0.56967, 0.99773), float3(0.25862, 0.57958, 0.99876), float3(0.25425, 0.58950, 0.99896), float3(0.24946, 0.59943, 0.99835), float3(0.24427, 0.60937, 0.99697), float3(0.23874, 0.61931, 0.99485), float3(0.23288, 0.62923, 0.99202), float3(0.22676, 0.63913, 0.98851), float3(0.22039, 0.64901, 0.98436), float3(0.21382, 0.65886, 0.97959), float3(0.20708, 0.66866, 0.97423), float3(0.20021, 0.67842, 0.96833), float3(0.19326, 0.68812, 0.96190), float3(0.18625, 0.69775, 0.95498), float3(0.17923, 0.70732, 0.94761), float3(0.17223, 0.71680, 0.93981), float3(0.16529, 0.72620, 0.93161), float3(0.15844, 0.73551, 0.92305), float3(0.15173, 0.74472, 0.91416), float3(0.14519, 0.75381, 0.90496), float3(0.13886, 0.76279, 0.89550), float3(0.13278, 0.77165, 0.88580), float3(0.12698, 0.78037, 0.87590), float3(0.12151, 0.78896, 0.86581), float3(0.11639, 0.79740, 0.85559), float3(0.11167, 0.80569, 0.84525), float3(0.10738, 0.81381, 0.83484), float3(0.10357, 0.82177, 0.82437), float3(0.10026, 0.82955, 0.81389), float3(0.09750, 0.83714, 0.80342), float3(0.09532, 0.84455, 0.79299), float3(0.09377, 0.85175, 0.78264), float3(0.09287, 0.85875, 0.77240), float3(0.09267, 0.86554, 0.76230), float3(0.09320, 0.87211, 0.75237), float3(0.09451, 0.87844, 0.74265), float3(0.09662, 0.88454, 0.73316), float3(0.09958, 0.89040, 0.72393), float3(0.10342, 0.89600, 0.71500), float3(0.10815, 0.90142, 0.70599), float3(0.11374, 0.90673, 0.69651), float3(0.12014, 0.91193, 0.68660), float3(0.12733, 0.91701, 0.67627), float3(0.13526, 0.92197, 0.66556), float3(0.14391, 0.92680, 0.65448), float3(0.15323, 0.93151, 0.64308), float3(0.16319, 0.93609, 0.63137), float3(0.17377, 0.94053, 0.61938), float3(0.18491, 0.94484, 0.60713), float3(0.19659, 0.94901, 0.59466), float3(0.20877, 0.95304, 0.58199), float3(0.22142, 0.95692, 0.56914), float3(0.23449, 0.96065, 0.55614), float3(0.24797, 0.96423, 0.54303), float3(0.26180, 0.96765, 0.52981), float3(0.27597, 0.97092, 0.51653), float3(0.29042, 0.97403, 0.50321), float3(0.30513, 0.97697, 0.48987), float3(0.32006, 0.97974, 0.47654), float3(0.33517, 0.98234, 0.46325), float3(0.35043, 0.98477, 0.45002), float3(0.36581, 0.98702, 0.43688), float3(0.38127, 0.98909, 0.42386), float3(0.39678, 0.99098, 0.41098), float3(0.41229, 0.99268, 0.39826), float3(0.42778, 0.99419, 0.38575), float3(0.44321, 0.99551, 0.37345), float3(0.45854, 0.99663, 0.36140), float3(0.47375, 0.99755, 0.34963), float3(0.48879, 0.99828, 0.33816), float3(0.50362, 0.99879, 0.32701), float3(0.51822, 0.99910, 0.31622), float3(0.53255, 0.99919, 0.30581), float3(0.54658, 0.99907, 0.29581), float3(0.56026, 0.99873, 0.28623), float3(0.57357, 0.99817, 0.27712), float3(0.58646, 0.99739, 0.26849), float3(0.59891, 0.99638, 0.26038), float3(0.61088, 0.99514, 0.25280), float3(0.62233, 0.99366, 0.24579), float3(0.63323, 0.99195, 0.23937), float3(0.64362, 0.98999, 0.23356), float3(0.65394, 0.98775, 0.22835), float3(0.66428, 0.98524, 0.22370), float3(0.67462, 0.98246, 0.21960), float3(0.68494, 0.97941, 0.21602), float3(0.69525, 0.97610, 0.21294), float3(0.70553, 0.97255, 0.21032), float3(0.71577, 0.96875, 0.20815), float3(0.72596, 0.96470, 0.20640), float3(0.73610, 0.96043, 0.20504), float3(0.74617, 0.95593, 0.20406), float3(0.75617, 0.95121, 0.20343), float3(0.76608, 0.94627, 0.20311), float3(0.77591, 0.94113, 0.20310), float3(0.78563, 0.93579, 0.20336), float3(0.79524, 0.93025, 0.20386), float3(0.80473, 0.92452, 0.20459), float3(0.81410, 0.91861, 0.20552), float3(0.82333, 0.91253, 0.20663), float3(0.83241, 0.90627, 0.20788), float3(0.84133, 0.89986, 0.20926), float3(0.85010, 0.89328, 0.21074), float3(0.85868, 0.88655, 0.21230), float3(0.86709, 0.87968, 0.21391), float3(0.87530, 0.87267, 0.21555), float3(0.88331, 0.86553, 0.21719), float3(0.89112, 0.85826, 0.21880), float3(0.89870, 0.85087, 0.22038), float3(0.90605, 0.84337, 0.22188), float3(0.91317, 0.83576, 0.22328), float3(0.92004, 0.82806, 0.22456), float3(0.92666, 0.82025, 0.22570), float3(0.93301, 0.81236, 0.22667), float3(0.93909, 0.80439, 0.22744), float3(0.94489, 0.79634, 0.22800), float3(0.95039, 0.78823, 0.22831), float3(0.95560, 0.78005, 0.22836), float3(0.96049, 0.77181, 0.22811), float3(0.96507, 0.76352, 0.22754), float3(0.96931, 0.75519, 0.22663), float3(0.97323, 0.74682, 0.22536), float3(0.97679, 0.73842, 0.22369), float3(0.98000, 0.73000, 0.22161), float3(0.98289, 0.72140, 0.21918), float3(0.98549, 0.71250, 0.21650), float3(0.98781, 0.70330, 0.21358), float3(0.98986, 0.69382, 0.21043), float3(0.99163, 0.68408, 0.20706), float3(0.99314, 0.67408, 0.20348), float3(0.99438, 0.66386, 0.19971), float3(0.99535, 0.65341, 0.19577), float3(0.99607, 0.64277, 0.19165), float3(0.99654, 0.63193, 0.18738), float3(0.99675, 0.62093, 0.18297), float3(0.99672, 0.60977, 0.17842), float3(0.99644, 0.59846, 0.17376), float3(0.99593, 0.58703, 0.16899), float3(0.99517, 0.57549, 0.16412), float3(0.99419, 0.56386, 0.15918), float3(0.99297, 0.55214, 0.15417), float3(0.99153, 0.54036, 0.14910), float3(0.98987, 0.52854, 0.14398), float3(0.98799, 0.51667, 0.13883), float3(0.98590, 0.50479, 0.13367), float3(0.98360, 0.49291, 0.12849), float3(0.98108, 0.48104, 0.12332), float3(0.97837, 0.46920, 0.11817), float3(0.97545, 0.45740, 0.11305), float3(0.97234, 0.44565, 0.10797), float3(0.96904, 0.43399, 0.10294), float3(0.96555, 0.42241, 0.09798), float3(0.96187, 0.41093, 0.09310), float3(0.95801, 0.39958, 0.08831), float3(0.95398, 0.38836, 0.08362), float3(0.94977, 0.37729, 0.07905), float3(0.94538, 0.36638, 0.07461), float3(0.94084, 0.35566, 0.07031), float3(0.93612, 0.34513, 0.06616), float3(0.93125, 0.33482, 0.06218), float3(0.92623, 0.32473, 0.05837), float3(0.92105, 0.31489, 0.05475), float3(0.91572, 0.30530, 0.05134), float3(0.91024, 0.29599, 0.04814), float3(0.90463, 0.28696, 0.04516), float3(0.89888, 0.27824, 0.04243), float3(0.89298, 0.26981, 0.03993), float3(0.88691, 0.26152, 0.03753), float3(0.88066, 0.25334, 0.03521), float3(0.87422, 0.24526, 0.03297), float3(0.86760, 0.23730, 0.03082), float3(0.86079, 0.22945, 0.02875), float3(0.85380, 0.22170, 0.02677), float3(0.84662, 0.21407, 0.02487), float3(0.83926, 0.20654, 0.02305), float3(0.83172, 0.19912, 0.02131), float3(0.82399, 0.19182, 0.01966), float3(0.81608, 0.18462, 0.01809), float3(0.80799, 0.17753, 0.01660), float3(0.79971, 0.17055, 0.01520), float3(0.79125, 0.16368, 0.01387), float3(0.78260, 0.15693, 0.01264), float3(0.77377, 0.15028, 0.01148), float3(0.76476, 0.14374, 0.01041), float3(0.75556, 0.13731, 0.00942), float3(0.74617, 0.13098, 0.00851), float3(0.73661, 0.12477, 0.00769), float3(0.72686, 0.11867, 0.00695), float3(0.71692, 0.11268, 0.00629), float3(0.70680, 0.10680, 0.00571), float3(0.69650, 0.10102, 0.00522), float3(0.68602, 0.09536, 0.00481), float3(0.67535, 0.08980, 0.00449), float3(0.66449, 0.08436, 0.00424), float3(0.65345, 0.07902, 0.00408), float3(0.64223, 0.07380, 0.00401), float3(0.63082, 0.06868, 0.00401), float3(0.61923, 0.06367, 0.00410), float3(0.60746, 0.05878, 0.00427), float3(0.59550, 0.05399, 0.00453), float3(0.58336, 0.04931, 0.00486), float3(0.57103, 0.04474, 0.00529), float3(0.55852, 0.04028, 0.00579), float3(0.54583, 0.03593, 0.00638), float3(0.53295, 0.03169, 0.00705), float3(0.51989, 0.02756, 0.00780), float3(0.50664, 0.02354, 0.00863), float3(0.49321, 0.01963, 0.00955), float3(0.47960, 0.01583, 0.01055) };
            float3 Turbo(float t)
            {
                int idx = clamp(int(255.0f * t), 0, 255);
                return turbo_srgb_floats[idx];
            }
            float3 viridis(float t) {

                const float3 c0 = float3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
                const float3 c1 = float3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
                const float3 c2 = float3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
                const float3 c3 = float3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
                const float3 c4 = float3(6.228269936347081, 14.17993336680509, 56.69055260068105);
                const float3 c5 = float3(4.776384997670288, -13.74514537774601, -65.35303263337234);
                const float3 c6 = float3(-5.435455855934631, 4.645852612178535, 26.3124352495832);

                return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));

            }

            float4 frag (v2f i) : SV_Target
            {
                // sample the texture
                float t = tex2D(_MainTex, i.uv).r;
                float4 col = 1;
                col.rgb = viridis(t);
                return col;
            }
            ENDCG
        }
    }
}