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