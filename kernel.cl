__kernel void marching_squares(
    __global const int* input_field,
    const int width, //output n_cols
    const int height, //output n_rows
    const int threshold,
    __global int* output_field
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int v0 = input_field[y * (width+1) + x];
    int v1 = input_field[y * (width+1) + (x + 1)];
    int v2 = input_field[(y + 1) * (width+1) + (x + 1)];
    int v3 = input_field[(y + 1) * (width+1) + x];

    // Determine the index for the edge table
    int square_index = 0;
    if (v0 > threshold) square_index |= 8;
    if (v1 > threshold) square_index |= 4;
    if (v2 > threshold) square_index |= 2;
    if (v3 > threshold) square_index |= 1;

    output_field[y * (width) + x] = square_index;
}

struct line_lookup
{
    int x0, y0, x1, y1, x2, y2, x3, y3;
};

__constant struct line_lookup lookup_table[] = {
    {-1, -1, -1, -1, -1, -1, -1, -1},//0
    {0, 1, 1, 2, -1, -1, -1, -1},//1
    {1, 2, 2, 1, -1, -1, -1, -1},//2
    {0, 1, 2, 1, -1, -1, -1, -1},//3
    {1, 0, 2, 1, -1, -1, -1, -1},//4
    {0, 1, 1, 0, 1, 2, 2, 1},//5
    {1, 0, 1, 2, -1, -1, -1, -1},//6
    {0, 1, 1, 0, -1, -1, -1, -1},//7
    {0, 1, 1, 0, -1, -1, -1, -1},//8
    {1, 0, 1, 2, -1, -1, -1, -1},//9
    {0, 1, 1, 2, 1, 0, 2, 1},//10
    {1, 0, 2, 1, -1, -1, -1, -1},//11
    {0, 1, 2, 1, -1, -1, -1, -1},//12
    {1, 2, 2, 1, -1, -1, -1, -1},//13
    {0, 1, 1, 2, -1, -1, -1, -1},//14
    {-1, -1, -1, -1, -1, -1, -1, -1}//15
};

void bresenhamLine(__global unsigned char* imageData, int width, int height, int x0, int y0, int x1, int y1) {
    if (x0 < 0)
    {
        return;
    }
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
  
    while (true) {
        int targetIndex = (y0 * width + x0);
        imageData[targetIndex] = 255; // Cast to Color* and assign
    
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

__kernel void draw_lines(
    __global const int* contour_values,
    const int input_width,
    const int input_height,
    __global unsigned char* output_image
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    struct line_lookup rel_coords = lookup_table[contour_values[y * input_width + x]];
    int x0 = x*5 + rel_coords.x0*2;
    int x1 = x*5 + rel_coords.x1*2;
    int x2 = x*5 + rel_coords.x2*2;
    int x3 = x*5 + rel_coords.x3*2;
    int y0 = y*5 + rel_coords.y0*2;
    int y1 = y*5 + rel_coords.y1*2;
    int y2 = y*5 + rel_coords.y2*2;
    int y3 = y*5 + rel_coords.y3*2;

    if (rel_coords.x0 >= 0)
    {
        bresenhamLine(output_image, input_width*5, input_height*5, x0, y0, x1, y1);
    }
    if (rel_coords.x2 >= 0)
    {
        bresenhamLine(output_image, input_width*5, input_height*5, x2, y2, x3, y3);
    }
}

