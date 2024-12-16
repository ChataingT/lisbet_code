import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt


def get_metrics(y_true, y_pred, idx_vid, mapping, out):
    tag = "by_frame"
    cr = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0)
    with open(os.path.join(out, f'best_classification_report_{tag}_over_test.txt'), 'w') as fd: 
        fd.write(cr)

    f1_best_fr = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average='weighted')

    tag= "avg_over_frame"
    dg = pd.DataFrame(data={'y_pred':y_pred, 'y_true':y_true, 'video':idx_vid})
    # ds = dg.groupby('video').std()

    dm = dg.groupby('video').mean()
    dm = dm.sort_values(by='y_true')
    dm = dm.reset_index()

    dm.y_pred = dm.y_pred.round()

    dm.y_true = dm.y_true.replace(mapping)
    dm.y_pred = dm.y_pred.replace(mapping)

    cr = classification_report(y_true=dm.y_true, y_pred=dm.y_pred, zero_division=0)
    with open(os.path.join(out, f'best_classification_report_{tag}_over_test.txt'), 'w') as fd: 
        fd.write(cr)

    f1_best_vid = f1_score(y_true=dm.y_true, y_pred=dm.y_pred, zero_division=0, average='weighted')

    return f1_best_fr, f1_best_vid
    
    

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.

    :param d: Dictionary to be flattened
    :param parent_key: Key of the parent, used during recursion
    :param sep: Separator for concatenating keys
    :return: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        new_key = new_key.replace(' ', '_')
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def debug_metrics(y_true, y_pred, idx_vid, mapping, epoch, out):
    tag = "by_frame"
    cr = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0, output_dict=True)
    cr = flatten_dict(cr)
    df = pd.DataFrame(cr, index=[epoch])
    
    df.to_csv(os.path.join(out, 'metrics_by_frame.csv'), header=(epoch == 0), mode='a+')

    tag= "avg_over_frame"
    dg = pd.DataFrame(data={'y_pred':y_pred, 'y_true':y_true, 'video':idx_vid})
    # ds = dg.groupby('video').std()

    dm = dg.groupby('video').mean()
    dm = dm.sort_values(by='y_true')
    dm = dm.reset_index()

    dm.y_pred = dm.y_pred.round()

    dm.y_true = dm.y_true.replace(mapping)
    dm.y_pred = dm.y_pred.replace(mapping)

    cr = classification_report(y_true=dm.y_true, y_pred=dm.y_pred, zero_division=0, output_dict=True)
    cr = flatten_dict(cr)
    df = pd.DataFrame(cr, index=[epoch])
    
    df.to_csv(os.path.join(out, 'metrics_avg_by_video.csv'), header=(epoch == 0), mode='a+')


def compute_validation(y_true, y_pred, videos, out, mapping):
    # prediction by frame
    dg = pd.DataFrame(data={'y_pred':y_pred, 'y_true':y_true, 'video':videos})
    dg.y_true = dg.y_true.replace(mapping)
    dg.y_pred = dg.y_pred.replace(mapping)

    tag = "by_frame"
    cr = classification_report(y_true=dg.y_true, y_pred=dg.y_pred)
    with open(os.path.join(out, f'classification_report_{tag}_over_val.txt'), 'w') as fd: 
        fd.write(cr)

    f1_best_fr = f1_score(y_true=dg.y_true, y_pred=dg.y_pred, zero_division=0, average='weighted')

    cm = confusion_matrix(y_true=dg.y_true, y_pred=dg.y_pred, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mapping.values())
    disp.plot()
    plt.savefig(os.path.join(out, f'confusion_matrix_{tag}_over_val.jpeg'))
    plt.close()


    # prediction with AVG over frame
    tag= "avg_over_frame"
    dg = pd.DataFrame(data={'y_pred':y_pred, 'y_true':y_true, 'video':videos})
    ds = dg.groupby('video').std()

    dm = dg.groupby('video').mean()
    dm = dm.sort_values(by='y_true')
    dm = dm.reset_index()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.scatter(x=dm.index.to_list(), y=dm.y_pred, c='r', label='Mean prediction', marker='o')
    plt.plot(dm.index.to_list(), dm.y_true, label=mapping, marker='x')
    plt.yticks([i/100 for i in range(0, 125, 25)])
    plt.grid()
    plt.title(f'Prediction of diagnosis based on classification over embedding frames')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, f'pred_diag_on_class_on_{tag}_over_val.jpg'), pad_inches=0.5)
    plt.close()

    dm.y_pred = dm.y_pred.round()

    dm.y_true = dm.y_true.replace(mapping)
    dm.y_pred = dm.y_pred.replace(mapping)

    cr = classification_report(y_true=dm.y_true, y_pred=dm.y_pred)
    with open(os.path.join(out, f'classification_report_{tag}_over_val.txt'), 'w') as fd: 
        fd.write(cr)

    cm = confusion_matrix(y_true=dm.y_true, y_pred=dm.y_pred, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mapping.values())
    disp.plot()
    plt.savefig(os.path.join(out, f'confusion_matrix_{tag}_over_val.jpeg'))
    plt.close()

    f1_best_vid = f1_score(y_true=dm.y_true, y_pred=dm.y_pred, zero_division=0, average='weighted')


    return f1_best_fr, f1_best_vid