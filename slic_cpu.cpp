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

struct  pixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

pixel* RGB_XYZ(pixel* img ,int ht ,int wd)
{
	for(int i=0; i<ht*wd;i++)
	{
		int R=img[i].r;
		int G=img[i].g;
		int B=img[i].b;

		float var_R=R/255;
		float var_G=G/255;
		float var_B=B/255;

		
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
	pixel *Pixel = (pixel*)malloc((img_ht)*(img_wd)*sizeof(pixel));
	
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
	pixel* Pixel_XYZ=(pixel*)malloc((img_ht*img_wd*sizeof(pixel));
	*Pixel_XYZ=RGB_XYZ(Pixel, img_ht, img_wd);
	
	//XYZ TO CIE-L*ab
	pixel* Pixel_LAB=(pixel*)malloc((img_ht*img_wd*sizeof(pixel));
	//*Pixel_LAB=XYZ_LAB(Pixel_XYZ, img_ht, img_wd)


	//OUTPUT STORAGE
	ofstream ofs;
	ofs.open("output.ppm", ofstream::out);
	ofs<<"P6\n"<<img_wd<<" "<<img_ht<<"\n"<<max_val<<"\n";

	for(int j=0; j <img_ht*img_wd;j++)
	{
		ofs<<Pixel_XYZ[j].r<<Pixel_XYZ[j].g<<Pixel_XYZ[j].b; //write as ascii
	}

	ofs.close();

	return 0;
}
