#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctime>

using namespace std;

struct pixel_RGB
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

struct pixel_XYZ
{
	float x;
	float y;
	float z;
};

pixel_XYZ* RGB_XYZ(pixel_RGB* img ,int ht ,int wd)
{	
	pixel_XYZ *XYZ=(pixel_XYZ*)(malloc(ht*wd*sizeof(pixel_XYZ)));

	for(int i=0; i<ht*wd;i++)
	{
		int R=img[i].r;
		int G=img[i].g;
		int B=img[i].b;

		float var_R=R/255;
		float var_G=G/255;
		float var_B=B/255;

		if(var_R>0.04045)
			var_R =pow((var_R + 0.055)/1.055,2.4);
		else                  
			 var_R = var_R / 12.92;

		if ( var_G > 0.04045 ) 
			var_G = pow((var_G + 0.055)/1.055,2.4);
		else                   
			var_G = var_G / 12.92;

		if ( var_B > 0.04045 ) 
			var_B = pow((var_B + 0.055 )/1.055, 2.4);
		else                   
			var_B = var_B / 12.92;

			var_R = var_R * 100;
			var_G = var_G * 100;
			var_B = var_B * 100;

		float X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
	    float Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
		float Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

		XYZ[i].x=X;
		XYZ[i].y=Y;
		XYZ[i].z=Z;
			
	}

	return XYZ;
}

pixel_XYZ* XYZ_LAB(pixel_XYZ* img ,int ht ,int wd)
{	
	pixel_XYZ *LAB_img=(pixel_XYZ*)(malloc(ht*wd*sizeof(pixel_XYZ)));

	for(int i=0; i<ht*wd;i++)
	{
		float X=img[i].x;
		float Y=img[i].y;
		float Z=img[i].z;

		float ref_X =  95.047;
		float ref_Y = 100.000;
		float ref_Z = 108.883;

		float var_X = X/ref_X;          //  Observer= 2°, Illuminant= D65
		float var_Y = Y/ref_Y;
		float var_Z = Z/ref_Z;          

		if(var_X > 0.008856) 
			var_X = pow(var_X,1/3);
		else                    
			var_X = (7.787*var_X ) + (16/116);

		if(var_Y > 0.008856) 
			var_Y = pow(var_Y,1/3);
		else                    
			var_Y = (7.787*var_Y) + (16/116);

		if(var_Z > 0.008856) 
			var_Z = pow(var_Z,1/3);
		else                    
			var_Z = (7.787*var_Z) + (16/116);

		float L = (116 * var_Y) - 16;
		float A = 500 * (var_X - var_Y);
		float B = 200 * (var_Y - var_Z);

		LAB_img[i].x=L;
		LAB_img[i].y=A;
		LAB_img[i].z=B;
	}
	return LAB_img;
}

int main(int argc, char* argv[])
{
	time_t start=time(NULL);
	if(argc != 2) //there should be three arguments
	return 1; //exit and return an error
	

	//READING FILE
	//reading the PPM file line by line
	
	ifstream infile;
	infile.open(argv[1]);
	string line;

	int img_wd, img_ht;
	int max_val;
	int line_count=0;

	//line one contains P6, line 2 mentions about gimp version, line 3 stores the height and width
	getline(infile, line);
	istringstream iss1(line);

	//reading first line to check format
	int word;
	string str1;
	iss1>>str1;
	
	if(str1.compare("P6")!=0) //comparing magic number
	{
		cout<<"wrong file format"<<endl;
		return 1;
	}
	
	getline(infile,line); //this line has version related comment, hence ignoring
	getline(infile,line); //this stores image dims

	istringstream iss2(line);
	iss2>>word;// this will be image width
	img_wd=word;
	iss2>>word;// this will be image height
	img_ht=word;
	
	cout<<img_ht<<" "<<img_wd<<endl;

	//storing the pixels as 1d images
	pixel_RGB *Pixel = (pixel_RGB*)malloc((img_ht)*(img_wd)*sizeof(pixel_RGB));
	
	int pix_cnt=0, cnt=0;

	getline(infile,line); //this stores max value
	istringstream iss3(line);
	iss3>>word;

	max_val=word;//max pixel value
	cout<<max_val<<endl;
	unsigned int val;

	while (getline(infile, line))
	{
		istringstream iss4(line);
		for (int i=0; i<=line.length();i++)
		{
			if(pix_cnt<img_ht*img_wd)
			{
				val =((int)line[i]);
				if(cnt%3==0)
				{
					Pixel[pix_cnt].r=val;
				}
				else if(cnt%3==1)
				{
					Pixel[pix_cnt].g=val;
				}
				else
				{
					Pixel[pix_cnt].b=val;
					pix_cnt++;
				}
				cnt++;
			}
		}
	}

	
	//COLOR CONVERSION
	//RGB->XYZ->CIE-L*ab

	//RGB to XYZ
	pixel_XYZ *Pixel_XYZ=RGB_XYZ(Pixel, img_ht, img_wd);
	
	//XYZ TO CIE-L*ab
	pixel_XYZ* Pixel_LAB=XYZ_LAB(Pixel_XYZ, img_ht, img_wd);


	//OUTPUT STORAGE
	ofstream ofs;
	ofs.open("output.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_val<<"\n";

	for(int j=0; j <img_ht*img_wd;j++)
	{
		ofs<<Pixel_LAB[j].x<<Pixel_LAB[j].y<<Pixel_LAB[j].z; //write as ascii
	}

	ofs.close();

	return 0;
}