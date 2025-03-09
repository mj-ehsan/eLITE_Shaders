///////////////PreProcessor-Definitions////
#define fov 70
#define IT_Intensity 0.99
#define UI_Saturation 1
///////////////PreProcessor-Definitions////
///////////////Include/////////////////////
#include "ReShade.fxh"
#include "CompleteFX_Common.fxh"
///////////////Include/////////////////////
///////////////PreProcessor-Definitions////

#define LDepth ReShade::GetLinearizedDepth
#define sTexColor ReShade::BackBuffer

#ifndef NGMV_DEBUG
 #define NGMV_DEBUG 0
#endif

#if __RENDERER__ > 0x9000
 #define NGMV_MAX_LEVEL 7
#else 
 #define NGMV_MAX_LEVEL 6
#endif

#define NGME_MAX_LOD (NGMV_MAX_LEVEL+1)
 
///////////////PreProcessor-Definitions////
///////////////Textures-Samplers///////////

texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler sTexMotionVectorsSampler { Texture = texMotionVectors; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

texture LBRTex  { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; MipLevels = NGME_MAX_LOD; };
sampler sLBRTex { Texture = LBRTex; };

#if __RENDERER__ > 0x9000
texture mv0Tex  { Width = BUFFER_WIDTH>>7; Height = BUFFER_HEIGHT>>7; Format = RG16f; };
texture f0Tex   { Width = BUFFER_WIDTH>>7; Height = BUFFER_HEIGHT>>7; Format = RG16f; };
sampler smv0Tex { Texture = mv0Tex;  };
sampler sf0Tex  { Texture = f0Tex; };
#endif

texture mv1Tex  { Width = BUFFER_WIDTH>>6; Height = BUFFER_HEIGHT>>6; Format = RG16f; };
texture mv2Tex  { Width = BUFFER_WIDTH>>5; Height = BUFFER_HEIGHT>>5; Format = RG16f; };
texture mv3Tex  { Width = BUFFER_WIDTH>>4; Height = BUFFER_HEIGHT>>4; Format = RG16f; };
texture mv4Tex  { Width = BUFFER_WIDTH>>3; Height = BUFFER_HEIGHT>>3; Format = RG16f; };
texture mv5Tex  { Width = BUFFER_WIDTH>>2; Height = BUFFER_HEIGHT>>2; Format = RG16f; };
texture mv6Tex  { Width = BUFFER_WIDTH>>1; Height = BUFFER_HEIGHT>>1; Format = RG16f; };
texture mv7Tex  { Width = BUFFER_WIDTH>>0; Height = BUFFER_HEIGHT>>0; Format = RG16f; };
sampler smv1Tex { Texture = mv1Tex; };
sampler smv2Tex { Texture = mv2Tex; };
sampler smv3Tex { Texture = mv3Tex; };
sampler smv4Tex { Texture = mv4Tex; };
sampler smv5Tex { Texture = mv5Tex; };
sampler smv6Tex { Texture = mv6Tex; };
sampler smv7Tex { Texture = mv7Tex; };

texture f1Tex   { Width = BUFFER_WIDTH>>6; Height = BUFFER_HEIGHT>>6; Format = RG16f; };
texture f2Tex   { Width = BUFFER_WIDTH>>5; Height = BUFFER_HEIGHT>>5; Format = RG16f; };
texture f3Tex   { Width = BUFFER_WIDTH>>4; Height = BUFFER_HEIGHT>>4; Format = RG16f; };
texture f4Tex   { Width = BUFFER_WIDTH>>3; Height = BUFFER_HEIGHT>>3; Format = RG16f; };
texture f5Tex   { Width = BUFFER_WIDTH>>2; Height = BUFFER_HEIGHT>>2; Format = RG16f; };
texture f6Tex   { Width = BUFFER_WIDTH>>1; Height = BUFFER_HEIGHT>>1; Format = RG16f; };
sampler sf1Tex  { Texture = f1Tex; };
sampler sf2Tex  { Texture = f2Tex; };
sampler sf3Tex  { Texture = f3Tex; };
sampler sf4Tex  { Texture = f4Tex; };
sampler sf5Tex  { Texture = f5Tex; };
sampler sf6Tex  { Texture = f6Tex; };

texture HistoryTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R8; MipLevels = 8; };
sampler sHistoryTex { Texture = HistoryTex; };
///////////////Textures-Samplers///////////
///////////////UI//////////////////////////

uniform int UI_QualityPreset <
	ui_label = "Preset";
	ui_type  = "combo";
	ui_items = "Performance\0Balance\0Quality\0";
> = 1;

int getShadingRate()
{
	if(UI_QualityPreset==0)return 3;
	if(UI_QualityPreset==1)return 2;
	if(UI_QualityPreset==2)return 1;
	return 0;
}

bool getUpscaleQuality()//true : performance mode
{
	if(UI_QualityPreset==0)return false;
	if(UI_QualityPreset==1)return false;
	if(UI_QualityPreset==2)return false;
	return true;
}

uniform float UI_FilterRadius <
	ui_label = "Filter Radius";
	ui_type  = "slider";
	ui_max   = 5;
	ui_step = 0.1;
	hidden   = true;
> = 2.5;

uniform int UI_Debug <
	ui_label = "Debug View";
	ui_type  = "combo";
	hidden   = true;
	ui_items = "MV\0"
	//"Error Delta\0Reference\0MV Length\0Accuracy\0"
	;
> = 0;

static const int   search_steps = 32;

///////////////UI//////////////////////////
///////////////Structures//////////////////
struct i
{
	float4 vpos : SV_Position;
	float2 uv : TexCoord0;
};
///////////////Structures//////////////////
///////////////Functions///////////////////
float get_lowest_level(){ return getShadingRate();}
///////////////Functions///////////////////
///////////////Pixel Shader////////////////
void prepass_PS(i i, out float2 outColor : SV_Target)
{
	outColor.x = lum(tex2D(sTexColor,i.uv).xyz);
	outColor.y = LDepth(i.uv);
}

#define CB_ARR_LEN 13
static const float2 block_kernel[CB_ARR_LEN] = 
{
	float2(0,0),//1
	float2( 1, 0),float2( 0, 1),float2(-1, 0),float2( 0,-1),//5
	float2( 2, 0),float2( 0, 2),float2(-2, 0),float2( 0,-2),//9
	float2( 2, 2),float2(-2,-2),float2( 2,-2),float2(-2, 2),//13
};

float2 filter(sampler hiMV, float2 uv)
{
	float c_depth = 0;
	c_depth = LDepth(uv);
	float w = 1, wsum;
	
	int counter = 0;
	float2 mean = 0;
	float2 mv_samples[9];
	float2 texelSize = rcp(tex2Dsize(hiMV)) * UI_FilterRadius;
	
	[unroll]for(int xx = -1; xx <= 1; xx++){
	[unroll]for(int yy = -1; yy <= 1; yy++)
	{
		float4 suv = float4(texelSize * float2(xx,yy) + uv, 0, 0);
		mv_samples[counter] = tex2Dlod(hiMV, suv).xy;
		w = exp(-abs(LDepth(suv.xy) - c_depth) * 20);
		mean += mv_samples[counter] * w;
		wsum += w;
		counter++;
	}}
	mean /= wsum;
	
	float min_dev = 1e+8;
	float2 result = 0;
	[unroll]for(int iter = 0; iter < 9; iter++)
	{
		float2 curr_mv = mv_samples[iter];
		float  dev = distance(curr_mv, mean);
		if(dev < min_dev)
		{
			min_dev = dev;
			result = curr_mv;
		}
	}
	return result;
}

float2 get_coarse_layer(sampler hiMV, float2 uv, float level)
{
	float2 plus_offsets[9] = 
	{
		0, 
		float2(0,-1),float2(-1,0),float2(1,0),float2(0, 1),
		float2(1, 1),float2(-1,1),float2(-1,-1),float2(1,-1)
	};
	//float2 plus_offsets[5] = {0, float2(1,-2),float2(-2,-1),float2(-1,2),float2(2,1)};
	float2 bm_width = exp2(level) * pix;
	
	float sumR = 0.0, sumSqR = 0.0;
    float sumC = 0.0, sumSqC = 0.0, sumRC = 0.0, sad = 0.0;
    float ivnBlockSize = rcp(9);
    
	float2 initMV = uv + tex2Dlod(hiMV, float4(uv,0,0)).xy;
	float  reference[5];
	for(int n; n < 5; n++)
	{
		reference[n] = tex2Dlod(sLBRTex, float4(block_kernel[n] * bm_width + uv,0,level)).x;
		
		sumR += reference[n];
		sumSqR += reference[n] * reference[n];
	}
	
	float meanR   = sumR   * ivnBlockSize;
	float meanSqR = sumSqR * ivnBlockSize;
	float varianceR = meanSqR - meanR * meanR;
		
	float2 ts = rcp(tex2Dsize(hiMV));
	ts *= max(1,UI_FilterRadius * 3);
	
	float max_NCC = -10, min_SAD = 1e+6;
	float2 out_mv = initMV - uv;
	float ties_num = 1;
	
	for(int n = 0; n < 5; n++)
	{
		float2 mv_uv = plus_offsets[n] * ts + uv;
		
		float2 block_mv = tex2Dlod(hiMV, float4(mv_uv,0,0)).xy;
		float2 block_uv = uv + block_mv;
		
		sumC = sumSqC = sumRC = sad = 0.0;
		for(int n2 = 0; n2 < 5; n2++)
		{
			float4 suv = float4(block_kernel[n2] * bm_width + block_uv,0,level);
			float candid = tex2Dlod(sHistoryTex, suv).x;
			
			sumC   += candid;
			sumSqC += candid * candid;
			sumRC  += reference[n2] * candid;
			
			sad += abs(reference[n2] - candid);
		}
	
		float meanC   = sumC   * ivnBlockSize;
		float meanSqC = sumSqC * ivnBlockSize;
		float meanRC  = sumRC * ivnBlockSize;
		
		float varianceC  = meanSqC - meanC * meanC;
		float covariance = meanRC - meanR * meanC;
		float NCC = covariance * rsqrt(varianceR * varianceC);
		
		if(NCC > max_NCC)
		{
			max_NCC = NCC;
			out_mv = block_mv;
			ties_num = 1;
		}
		else if(NCC == max_NCC)
		{
			out_mv += block_mv;
			ties_num++;
		}
	}
	
	return out_mv / ties_num;
}

float3 get_candid(in float2 block_uv, float2 grad_offset[2], float level)
{
	float3 c;
	c.x = tex2Dlod(sHistoryTex, float4(block_uv, 0, level)).x;
	c.y = tex2Dlod(sHistoryTex, float4(block_uv + grad_offset[0], 0, level)).x;
	c.z = tex2Dlod(sHistoryTex, float4(block_uv + grad_offset[1], 0, level)).x;

	return c;
}

//Linearized Adam Gradient Ascend Normalized crOss cOrrelation Block mAtching
float2 LAGANOOBA(in float2 uv, in int level, in sampler hiMVsamp, out float2 debugView)
{
	float invArrLength = rcp(CB_ARR_LEN);
	float2 initMV = uv;
	if(level < NGMV_MAX_LEVEL)
		initMV = get_coarse_layer(hiMVsamp, uv, level)+uv;
	
	static const float2 size = exp2(level) * pix;
	
    float sumR = 0.0, sumSqR = 0.0;
    float3 sumC = 0.0, sumSqC = 0.0, sumRC = 0.0;
    float2 grad_k[2] = {float2(size.x*pix.x*10,0),float2(0,size.y*pix.y*10)};
	float reference[CB_ARR_LEN];
	[unroll]for(int k = 0; k < CB_ARR_LEN; k++)
	{
		reference[k] = tex2Dlod(sLBRTex, float4(size * block_kernel[k] + uv,0,level)).x;
		float3 candid = get_candid(size * block_kernel[k] + initMV, grad_k, level);
		
		sumR += reference[k];
		sumSqR += reference[k] * reference[k];
		
		sumC   += candid;
		sumSqC += candid * candid;
		sumRC  += reference[k] * candid;
	}
	float  meanR = sumR * invArrLength;
	float3 meanC = sumC * invArrLength;
	
	float  varianceR = sumSqR * invArrLength - meanR * meanR;
    float3 varianceC = sumSqC * invArrLength - meanC * meanC;
	varianceR = max(varianceR, 1e-6);
	varianceC = max(varianceC, 1e-6);

    float3 covariance = sumRC * invArrLength - meanR * meanC;
	float3 NCC = covariance * rsqrt(varianceR * varianceC);
	
	
	
	float3 gradient_avg;
	float3 gradient_offset;
	
	gradient_offset.xy = (NCC.yz - NCC.xx) / float2(grad_k[0].x, grad_k[1].y);
	gradient_offset.z  = dot(gradient_offset.xy, gradient_offset.xy);
	float3 gradient_sum = gradient_offset;
	gradient_avg = gradient_offset;
	
	gradient_offset.xy = gradient_avg.xy * rsqrt(gradient_avg.z + 1e-6);
	float2 search_uv = gradient_offset.xy * pix;
	
	float2 best_match = 0;
	float  max_NCC = NCC.x;
	float  num_ties = 1;
	
	for(int iter = 0; iter < search_steps; iter++)
	{
		sumC = 0.0, sumSqC = 0.0, sumRC = 0.0;
		float2 block_uv = search_uv + initMV;
		if(block_uv.x <= 0 || block_uv.x >= 1 || block_uv.y <= 0 || block_uv.y >= 1) continue;
		
		[unroll]for(int k = 0; k < CB_ARR_LEN; k++)
		{
			float2 suv = size * block_kernel[k] + block_uv;
			float3 candid = get_candid(suv, grad_k, level);
			
			sumC   += candid;
			sumSqC += candid * candid;
			sumRC  += reference[k] * candid;
		}
		
	    meanC = sumC * invArrLength;
	    varianceC = sumSqC * invArrLength - meanC * meanC;
		varianceC = max(varianceC, 1e-6);
		
	    float3 covariance = sumRC * invArrLength - meanR * meanC;
		float3 NCC = covariance * rsqrt(varianceR * varianceC);
	
		if(NCC.x > max_NCC)
		{
			max_NCC = NCC.x;
			best_match = search_uv;
			num_ties = 1;
		}
		else break;
		
		gradient_offset.xy = (NCC.yz - NCC.xx) / float2(grad_k[0].x, grad_k[1].y);
		gradient_offset.z  = dot(gradient_offset.xy, gradient_offset.xy);
		gradient_sum += gradient_offset;
		gradient_avg = gradient_sum / (2+iter);
		
		gradient_offset.xy = gradient_avg.xy * rsqrt(gradient_avg.z + 1e-8);
		gradient_offset.xy *= pix;
		
		float gmc = min(abs(gradient_offset.x) / pix.x, abs(gradient_offset.y) / pix.y);
		if(gmc < 0.1)break;
		
		search_uv += gradient_offset.xy;
	}
	best_match /= num_ties;
	
	debugView = max_NCC.xx;
    return best_match.xy + initMV - uv;
}

float2 calculateMV(in float2 uv, in int level, in sampler hiMVsamp)
{
	float2 debugView;
	float2 mv = LAGANOOBA(uv, level, hiMVsamp, debugView);
	
	if(UI_Debug == 1 && level <= get_lowest_level())
		return debugView.xy;
	else
		return mv;
		
	return 0;
}

float2 getMV(in float2 uv, in int level, in sampler hiMVsamp)
{
	if(level < get_lowest_level())
		if(UI_Debug == 1) return tex2D(hiMVsamp, uv).xy;
		else
			if(getUpscaleQuality()) 
				return filter(hiMVsamp, uv);
			else
				return get_coarse_layer(hiMVsamp, uv, level);
	else
		return calculateMV(uv, level, hiMVsamp);
	
	return 0;
}

#if __RENDERER__ > 0x9000
float2 mv_level0(i i):SV_Target {return getMV(i.uv, 7, sf1Tex);}
float2 mv_level1(i i):SV_Target {return getMV(i.uv, 6, sf0Tex);}
float2 f_level0(i i):SV_Target {return filter(smv0Tex, i.uv);}
#else
float2 mv_level1(i i):SV_Target {return getMV(i.uv, 6, sf2Tex);}
#endif
float2 mv_level2(i i):SV_Target {return getMV(i.uv, 5, sf1Tex);}
float2 mv_level3(i i):SV_Target {return getMV(i.uv, 4, sf2Tex);}
float2 mv_level4(i i):SV_Target {return getMV(i.uv, 3, sf3Tex);}
float2 mv_level5(i i):SV_Target {return getMV(i.uv, 2, sf4Tex);}
float2 mv_level6(i i):SV_Target {return getMV(i.uv, 1, sf5Tex);}
float2 mv_level7(i i):SV_Target {return getMV(i.uv, 0, sf6Tex);}

float2 f_level1(i i):SV_Target {return filter(smv1Tex, i.uv);}
float2 f_level2(i i):SV_Target {return filter(smv2Tex, i.uv);}
float2 f_level3(i i):SV_Target {return filter(smv3Tex, i.uv);}
float2 f_level4(i i):SV_Target {return filter(smv4Tex, i.uv);}
float2 f_level5(i i):SV_Target {return filter(smv5Tex, i.uv);}
float2 f_level6(i i):SV_Target {return filter(smv6Tex, i.uv);}

float  history_PS(i i) : SV_Target {return tex2D(sLBRTex, i.uv).x;}

float3 HUEtoRGB(in float H)
{
	float R = abs(H * 6.f - 3.f) - 1.f;
	float G = 2 - abs(H * 6.f - 2.f);
	float B = 2 - abs(H * 6.f - 4.f);
	return saturate(float3(R,G,B));
}

float3 HSLtoRGB(in float3 HSL)
{
	float3 RGB = HUEtoRGB(HSL.x);
	float C = (1.f - abs(2.f * HSL.z - 1.f)) * HSL.y;
	return (RGB - 0.5f) * C + HSL.z;
}

float3 motionDebug(float2 motion)
{
	float angle = degrees(atan2(motion.y, motion.x));
	float dist = length(motion);
	float3 rgb = HSLtoRGB(float3((angle / 360.f) + 0.5, saturate(dist * 100.0), 0.5));
	return rgb;
}

float3 debugOutput_PS(i i) : SV_Target
{
	float2 mv = filter(smv7Tex, i.uv).xy;
	
	if(!NGMV_DEBUG) return mv.xyx;
	
	if(UI_Debug == 0)
		return motionDebug(tex2D(smv7Tex, i.uv).xy);
	else if(UI_Debug == 1)
		return tex2D(smv7Tex, i.uv).xyz;
	else if(UI_Debug == 2)
		return tex2D(sLBRTex, i.uv).x;
	else if(UI_Debug == 3)
		return length(tex2D(smv7Tex, i.uv).xy)*100;
	else if(UI_Debug == 4)
	{
		float sumR, sumH, sumRH, sumSqR, sumSqH;
		for(int xx = -2; xx <= 2; xx++){
		for(int yy = -2; yy <= 2; yy++)
		{
			float r = tex2Dlod(sLBRTex,     float4(float2(xx,yy) * BUFFER_PIXEL_SIZE + i.uv, 0, 0)).x;
			float h = tex2Dlod(sHistoryTex, float4(float2(xx,yy) * BUFFER_PIXEL_SIZE + i.uv + mv, 0, 0)).x;
			
			sumR += r;
			sumH += h;
			sumRH += r*h;
			sumSqR += r*r;
			sumSqH += h*h;
		}}
		
		float wnorm = rcp(25.);
		sumR *= wnorm;
		sumH *= wnorm;
		float varianceR = sumSqR * wnorm - sumR * sumR;
		float varianceH = sumSqH * wnorm - sumH * sumH;
		
		float covariance = sumRH * wnorm - sumR * sumH;
		float normalizedCrossCorrelation = covariance * rsqrt(varianceR * varianceH + 1e-6);
		
		return (normalizedCrossCorrelation) >= 0;
	}
	return 0;
}
///////////////Pixel Shader////////////////

#define MV_PASS(PS_NAME, TEX_NAME) \
	pass \
    { \
    	VertexShader  = PostProcessVS; \
    	PixelShader   = PS_NAME; \
    	RenderTarget  = TEX_NAME; \
    }
    
#define FILTER MV_PASS

technique eLITE_Motion <
    ui_label = "eLITE Motion";
	ui_tooltip = "||          eLITE Motion || Version 1.0.0         ||\n"
	             "||                   By NiceGuy                   ||\n"
	             "||High Performance High Quality Motion Estimation.||";
>
{
    pass
    {
    	VertexShader  = PostProcessVS;
    	PixelShader   = prepass_PS;
    	RenderTarget0 = LBRTex;
    }
#if __RENDERER__ > 0x9000
    MV_PASS(mv_level0, mv0Tex)
    FILTER(f_level0, f0Tex)
#endif
    MV_PASS(mv_level1, mv1Tex)
    FILTER(f_level1, f1Tex)
    MV_PASS(mv_level2, mv2Tex)
    FILTER(f_level2, f2Tex)
    MV_PASS(mv_level3, mv3Tex)
    FILTER(f_level3, f3Tex)
    MV_PASS(mv_level4, mv4Tex)
    FILTER(f_level4, f4Tex)
    MV_PASS(mv_level5, mv5Tex)
    FILTER(f_level5, f5Tex)
    MV_PASS(mv_level6, mv6Tex)
    FILTER(f_level6, f6Tex)
    MV_PASS(mv_level7, mv7Tex)
    pass
    {
    	VertexShader  = PostProcessVS;
    	PixelShader   = debugOutput_PS;
#if !NGMV_DEBUG
		RenderTarget  = texMotionVectors;
#endif
    }
    pass
    {
    	VertexShader  = PostProcessVS;
    	PixelShader   = history_PS;
    	RenderTarget  = HistoryTex;
    }
}