use polars::prelude::*;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;
use smartcore::readers::csv;
use smartcore::readers::csv::matrix_from_csv_source;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::ops::Range;

fn main() -> Result<(), Box<dyn Error>> {
    let data = load_data().expect("Erro ao carregar os dados");
    train_model(data)?;
    //     let model = load_model()?;
    //     let r = model.predict(&DenseMatrix::new(1, 1, vec![2.0], false))?;
    //     println!("Predictions: {:?}", r);
    Ok(())
}

fn load_data() -> Result<((Vec<f64>, Vec<f64>)), Box<dyn Error>> {
    // Leia o arquivo CSV em um DataFrame
    let df = CsvReadOptions::default()
        .with_infer_schema_length(None)
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("pontuacao_teste.csv".into()))?
        .finish()?;
    println!("{}", df);

    let mx_vec: Vec<f64> = df
        .column("horas_estudo")
        .unwrap()
        .f64()
        .unwrap()
        .cont_slice()
        .unwrap()
        .to_vec();

    let my_vec: Vec<f64> = df
        .column("pontuacao_teste")
        .unwrap()
        .f64()
        .unwrap()
        .cont_slice()
        .unwrap()
        .to_vec();

    Ok((mx_vec, my_vec))
}

fn train_model(data: (Vec<f64>, Vec<f64>)) -> Result<(), Box<dyn Error>> {
    let (mx_vec, my_vec) = data;
    let mx = DenseMatrix::new(mx_vec.len(), 1, mx_vec, false);

    let my = DenseMatrix::new(my_vec.len(), 1, my_vec, false);

    // Create a Linear Regression model
    let model = LinearRegression::fit(&mx, &my, Default::default())?;

    // y = ax + b
    //     let intercept = model.intercept();
    //     let coefficients = model.coefficients();
    //     let cof = coefficients.get((0, 0));
    //     println!("y = {}x + {}", cof, intercept);

    //     Predict the value of y for a new inputs
    //     let inputs = DenseMatrix::new(5, 1, vec![2.0, 4.0, 6.0, 8.0, 10.0], false);
    //     let total_predictions = model.predict(&inputs)?;
    //     println!("Total Predictions: {:?}", total_predictions);

    //     Predict the value of y for a new input = 2.0
    //     let input = DenseMatrix::new(1, 1, vec![2.0], false);
    //     let y_pred = model.predict(&input)?;
    //     println!("Predictions: {:?}", y_pred);
    //     // salvar o modelo treinado
    //     save_model(&model)
    Ok(())
}

type LinearRegressionModel = LinearRegression<f64, i32, DenseMatrix<f64>, Vec<i32>>;

fn save_model(model: &LinearRegressionModel) -> Result<(), Box<dyn Error>> {
    let serialized_model = serde_json::to_string(&model)?;
    let mut file = File::create("model.json")?;
    file.write_all(serialized_model.as_bytes())?;
    Ok(())
}

fn load_model() -> Result<LinearRegressionModel, Box<dyn Error>> {
    let file = File::open("model.json")?;
    let model: LinearRegressionModel = serde_json::from_reader(file).unwrap();
    Ok(model)
}

// let data = vec![vec![1.0], vec![3.0], vec![5.0], vec![7.0], vec![9.0]];

// let x = DenseMatrix::from_2d_vec(&data);
// let file = fs::File::open("pontuacao_teste.csv").unwrap();

// let mx: DenseMatrix<f64> = csv::matrix_from_csv_source::<f64, Vec<_>, DenseMatrix<_>>(
//     file,
//     csv::CSVDefinition::default(),
// )
// .unwrap();
