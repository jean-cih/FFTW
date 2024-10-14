#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include </usr/local/include/fftw3.h>

#define N 512
#define M 512

#define GAP_WIDTH 20
#define GAP_HEIGHT 5

#define REAL 0
#define IMAG 1

#define PI 3.14159265358979323846

#define FOCAL 8000      //mm
#define ENERGY 30       //keV
#define WL 1e-6 * 1239.8 / (ENERGY * 1e3) //mm
#define DELTA_AL 5.43e-4 / (ENERGY * ENERGY)
#define MU_AL 3.0235443337462318        // 30 keV
#define BETA_AL MU_AL * WL / (4 * PI)       // 30 keV
                                



double func_exp(double x, double y, double z);
double func_exp_ft(double x, double y);
double func_energy_real(double x, double y, double z);
double func_energy_imag(double x, double y, double z);
double func_sqrt(double x, double y, double z);
double func_optic_t(double y);

double find_max(fftw_complex *data, int flag);

void spectrum_save_png(const char* filename, int width, int height, unsigned char* data);
void shift_spectrum(unsigned char *spectrumOut, unsigned char *reverseSpectrumOut);

void output_data(fftw_complex *in, fftw_complex *out, fftw_complex *back);


int main() {
    // Создание массивов для FFTW
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);
    fftw_complex *back = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * M);

    // Создание плана для прямого преобразования Фурье
    fftw_plan plan_forward = fftw_plan_dft_2d(N, M, in, out, FFTW_FORWARD, FFTW_ESTIMATE);


    // Заполнение входных данных функцией сигналов
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            double fx = (double)i - N / 2;
            double fy = (double)j - M / 2;
            //double fx = (double)i / N;
            //double fy = (double)j / M;      
            
            if(fabs(fx) < GAP_HEIGHT && fabs(fy) < GAP_WIDTH){
                in[i * N + j][REAL] = 1.0;// * exp(PI * (fx * fx + fy * fy));
            }
            else{
                in[i * N + j][REAL] = 0.0;
            }
            
            //in[i * N + j][REAL] = func_energy_real(fx, fy, 0) * cos(func_exp_ft(fx, fy));
            //in[i * N + j][IMAG] = func_energy_imag(fx, fy, 0) * sin(func_exp_ft(fx, fy));

            //in[i * N + j][REAL] = cos(func_exp_ft(fx, fy));
            //in[i * N + j][IMAG] = sin(func_exp_ft(fx, fy));
        }
    }


    double maxInReal = find_max(in, REAL);
    double maxInImag = find_max(in, IMAG);

    double maxIn = sqrt(pow(maxInReal, 2) + pow(maxInImag, 2));

    // Массив для хранения амплитуд спектра
    unsigned char* spectrumInp = (unsigned char*)malloc(sizeof(unsigned char) * N * M);

    // Вычисление амплитуд спектра
    for (int i = 0; i < N * M; i++) {
        spectrumInp[i] = (unsigned char)(sqrt(pow(in[i][REAL], 2) + pow(in[i][IMAG], 2)) * 255 / maxIn);
    }

    // Сохранение результата в PNG файл
    spectrum_save_png("inputSpectrum.png", N, M, spectrumInp);

    // Выполнение прямого преобразования Фурье
    fftw_execute(plan_forward);

    
    double maxOutReal = find_max(out, REAL);
    double maxOutImag = find_max(out, IMAG);

    double maxOut = sqrt(pow(maxOutReal, 2) + pow(maxOutImag, 2));

    // Массив для хранения амплитуд спектра
    unsigned char* spectrumOut = (unsigned char*)malloc(sizeof(unsigned char) * N * M);

    // Вычисление амплитуд спектра
    for (int i = 0; i < N * M; i++){
        spectrumOut[i] = (unsigned char)(sqrt(pow(out[i][REAL], 2) + pow(out[i][IMAG], 2)) * 255 / maxOut);
    }
    
    unsigned char* reverseSpectrumOut = (unsigned char*)malloc(sizeof(unsigned char) * N * M);

    shift_spectrum(spectrumOut, reverseSpectrumOut);

    // Сохранение результата в PNG файл
    spectrum_save_png("outputSpectrum.png", N, M, reverseSpectrumOut);


    fftw_plan plan_backward = fftw_plan_dft_2d(N, M, out, back, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Выполнение преобразования
    fftw_execute(plan_backward);

    // Массив для хранения амплитуд спектра
    unsigned char* spectrumBack = (unsigned char*)malloc(sizeof(unsigned char) * N * M);

    // Вычисление амплитуд спектра
    for (int i = 0; i < N * M; i++){
        spectrumBack[i] = (unsigned char)(sqrt(back[i][REAL] * back[i][REAL] + back[i][IMAG] * back[i][IMAG]) / N / M * 255);
    }

    // Сохранение результата в PNG файл
    spectrum_save_png("backSpectrum.png", N, M, spectrumBack);

    // Вывод результата
    output_data(in, out, back);

    // Освобождение памяти
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(in);
    fftw_free(out);
    fftw_free(back);
    free(spectrumInp);
    free(spectrumOut);
    free(spectrumBack);
    free(reverseSpectrumOut);

    return 0;
}

double func_exp(double x, double y, double z){
    return exp(2 * PI * (z + (x * x + y * y) / (2 * z)));
}

double func_exp_ft(double x, double y){
    return exp(PI * (x * x + y * y));
}

double func_energy_real(double x, double y, double z) {
    return (cos(func_sqrt(x, y, z)) / func_sqrt(x, y, z)) * cos(-(DELTA_AL - BETA_AL) * func_optic_t(y)); 
}

double func_energy_imag(double x, double y, double z) {
    return (sin(func_sqrt(x, y, z)) / func_sqrt(x, y, z)) * sin(-(DELTA_AL - BETA_AL) * func_optic_t(y)); 
}

double func_sqrt(double x, double y, double z){
    return sqrt(x * x + y * y + z * z);
}

double func_optic_t(double y){
    return y * y / (2 * FOCAL * DELTA_AL);
}

double find_max(fftw_complex *data, int flag){

    double max = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if(data[i * N + j][flag] > max) {
                max = data[i * N + j][flag];
            }
        }
    }

    return max;
}

void shift_spectrum(unsigned char *spectrumOut, unsigned char *reverseSpectrumOut){

    int j = 0, k = 0, l = 0, t = 0;
    for(int i = 0; i < N * M / 2; i++){

        if(i % (N / 2) == 0){
            i += N / 2;
        }

        reverseSpectrumOut[j] = spectrumOut[N * M / 2 - 1 + i];
        reverseSpectrumOut[i] = spectrumOut[N * M / 2 - 1 + k];
        reverseSpectrumOut[N * M / 2 + 1 + i] = spectrumOut[l];
        reverseSpectrumOut[N * M / 2 + 1 + t] = spectrumOut[i];

        j++;
        k++;
        l++;
        t++;

        if(j % (N / 2) == 0){
            j += N / 2;
        }
        if(k % (N / 2) == 0){
            k += N / 2;
        }
        if(l % (N / 2) == 0){
            l += N / 2;
        }
        if(t % (N / 2) == 0){
            t += N / 2;
        }
    }
}


void spectrum_save_png(const char* filename, int width, int height, unsigned char* data) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Ошибка открытия файла: %s\n", filename);
        return;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Ошибка создания PNG структуры\n");
        fclose(fp);
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Ошибка создания PNG информации\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Ошибка инициализации PNG\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = data + y * width;
    }

    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, info_ptr);

    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    printf("png файл сохранен\n");
}

void output_data(fftw_complex *in, fftw_complex *out, fftw_complex *back){

    for (int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            double amp_in = sqrt(in[i * N + j][REAL] * in[i * N + j][REAL] + in[i * N + j][IMAG] * in[i * N + j][IMAG]); // Амплитуда in
            double amp_out = sqrt(out[i * N + j][REAL] * out[i * N + j][REAL] + out[i * N + j][IMAG] * out[i * N + j][IMAG]); // Амплитуда out
            double amp_back = sqrt(back[i * N + j][REAL] * back[i * N + j][REAL] + back[i * N + j][IMAG] * back[i * N + j][IMAG]) / N / M; // Амплитуда back

            double phase_in = atan2(in[i * N + j][IMAG], in[i * N + j][REAL]); // Фаза in
            double phase_out = atan2(out[i * N + j][IMAG], out[i * N + j][REAL]); // Фаза out
            double phase_back = atan2(back[i * N + j][IMAG], back[i * N + j][REAL]); // Фаза back

            if(amp_in == 1){
                printf("Амплитуда in %.2f  Фаза in %.2f | Амплитуда out %.2f  Фаза out %.2f | Амплитуда back %.2f  Фаза back %.2f\n", amp_in, phase_in, amp_out, phase_out, amp_back, phase_back);
            }
        }
    }
}
