use std::{
    fmt::Display,
    process::exit,
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use gen::image::*;
use rand::Rng;
use tui::{
    style::*,
    text::{Span, Spans},
    widgets::*,
};

mod gen;

fn main() {
    let start_time = chrono::offset::Utc::now();

    let promts = vec![
        // "photo of the early universe",
        "photo of the early morning hours",
        // "photo of the earth from space",
        "photo of mountains",
        "surreal landscape",
        // "futuristic technology",
        // "image of a mythical creature that has never been seen before",
        "image of an alien planet with unique flora and fauna",
        // "cityscape that merges different architectural styles",
        // "futuristic vehicle",
    ];

    let (tx, rx): (Sender<Info>, Receiver<Info>) = mpsc::channel();

    thread::spawn(move || -> Result<(), anyhow::Error> {
        use crossterm::{
            event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
            execute,
            terminal::{
                disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
            },
        };
        use std::{io, thread, time::Duration};
        use tui::{
            backend::CrosstermBackend,
            layout::{Constraint, Direction, Layout},
            widgets::{Block, Borders, Widget},
            Terminal,
        };

        // enable_raw_mode()?;
        // let mut stdout = io::stdout();
        // execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        // let backend = CrosstermBackend::new(stdout);
        // let mut terminal = Terminal::new(backend)?;

        let mut cuda = false;
        let mut cudnn = false;
        let mut mps = false;

        let mut workload: Option<Workload> = None;

        let mut step: usize = 0;

        let mut image_duration = Duration::from_secs(0);
        let mut timestep_duration = Duration::from_secs(0);

        let mut image_arg: Option<Args> = None;

        println!("");

        loop {
            match rx.recv() {
                Ok(info) => {
                    match info.clone() {
                        Info::System {
                            cuda: cuda_info,
                            cudnn: cudnn_info,
                            mps: mps_info,
                        } => {
                            cuda = cuda_info;
                            cudnn = cudnn_info;
                            mps = mps_info;
                        }
                        Info::Building(w) => {
                            workload = Some(w);
                        }
                        Info::TimestepStart(n) => {
                            step = n;
                        }
                        Info::TimestepDone(d) => {
                            timestep_duration = d;
                            print!(
                                "\rImage: {: >4}ms Timestep: {: >3}ms {: >2}",
                                image_duration.as_millis(),
                                timestep_duration.as_millis(),
                                step,
                            );
                        }
                        Info::ImageStart(a) => {
                            image_arg = Some(a);
                        }
                        Info::ImageDone(d) => {
                            image_duration = d;
                        }
                        Info::Done => break,
                    }

                    // terminal.draw(|f| {
                    //     let size = f.size();
                    //     let seed = match image_arg {
                    //         Some(ref a) => a.seed.to_string(),
                    //         None => "".to_string(),
                    //     };
                    //     f.render_widget(
                    //         Table::new(vec![
                    //             Row::new(vec![
                    //                 format!("{}", state),
                    //                 format!("{}", step.to_string()),
                    //                 format!("{}", timestep_duration.as_millis()),
                    //             ]),
                    //             Row::new(vec![
                    //                 format!("{}", seed),
                    //                 format!("{}", ""),
                    //                 format!("{}", image_duration.as_millis()),
                    //             ]),
                    //         ])
                    //         .block(Block::default().title("Table").borders(Borders::ALL)),
                    //             size,
                    //         );
                    // })?;

                    if let Info::Done = info {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        // disable_raw_mode()?;
        // execute!(
        //     terminal.backend_mut(),
        //     LeaveAlternateScreen,
        //     DisableMouseCapture
        // )?;
        // terminal.show_cursor()?;

        Ok(())
    });

    let mut rng = rand::thread_rng();

    let _ = gen::image::batch_default(
        (0..2)
            .map(|_| {
                let seed = rng.gen::<i64>();

                promts.iter().map(move |prompt| {
                    let folder =
                        std::path::PathBuf::from(format!("out/[{:?}]/({})", start_time, prompt));
                    std::fs::create_dir_all(&folder).unwrap();

                    let path = folder.join(format!("({:x}).png", seed));

                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent).unwrap();
                    }

                    Args {
                        prompt: prompt.to_string(),
                        seed,
                        path,
                    }
                })
            })
            .flatten()
            .collect::<Vec<_>>(),
        tx,
    )
    .unwrap();

    loop {}
}
