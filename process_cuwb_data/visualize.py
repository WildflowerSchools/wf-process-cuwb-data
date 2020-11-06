import numpy as np
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import slugify
import os

register_matplotlib_converters()


def plot_positions_multiple_devices(
    df,
    room_corners,
    **kwargs
):
    for (device_id, device_serial_number, entity_type, person_short_name, material_name), group_df in df.fillna('NA').groupby([
        'device_id',
        'device_serial_number',
        'entity_type',
        'person_short_name',
        'material_name'
    ]):
        entity_name = material_name
        if entity_type == 'Person':
            entity_name = person_short_name
        plot_positions(
            df=group_df,
            entity_name=entity_name,
            room_corners=room_corners,
            device_serial_number=device_serial_number,
            **kwargs
        )


def plot_positions_topdown_multiple_devices(
    df,
    room_corners,
    **kwargs
):
    for (device_id, device_serial_number, entity_type, person_short_name, material_name), group_df in df.fillna('NA').groupby([
        'device_id',
        'device_serial_number',
        'entity_type',
        'person_short_name',
        'material_name'
    ]):
        entity_name = material_name
        if entity_type == 'Person':
            entity_name = person_short_name
        plot_positions_topdown(
            df=group_df,
            entity_name=entity_name,
            room_corners=room_corners,
            device_serial_number=device_serial_number,
            **kwargs
        )


def plot_accelerations_multiple_devices(
    df,
    **kwargs
):
    for (device_id, device_serial_number, entity_type, person_short_name, material_name), group_df in df.fillna('NA').groupby([
        'device_id',
        'device_serial_number',
        'entity_type',
        'person_short_name',
        'material_name'
    ]):
        entity_name = material_name
        if entity_type == 'Person':
            entity_name = person_short_name
        plot_accelerations(
            df=group_df,
            entity_name=entity_name,
            device_serial_number=device_serial_number,
            **kwargs
        )


def plot_positions_and_accelerations_multiple_devices(
    df_position,
    df_acceleration,
    room_corners,
    **kwargs
):
    for (device_id, device_serial_number, entity_type, person_short_name, material_name), df_position_group_df in df_position.fillna('NA').groupby([
        'device_id',
        'device_serial_number',
        'entity_type',
        'person_short_name',
        'material_name'
    ]):
        entity_name = material_name
        if entity_type == 'Person':
            entity_name = person_short_name
        df_acceleration_group_df = df_acceleration.loc[df_acceleration['device_id'] == device_id]
        plot_positions_and_accelerations(
            df_position=df_position_group_df,
            df_acceleration=df_acceleration_group_df,
            entity_name=entity_name,
            device_serial_number=device_serial_number,
            room_corners=room_corners,
            **kwargs
        )


def plot_tray_motion_features_multiple_devices(
    df_position,
    df_features,
    room_corners=None,
    velocity_limits=None,
    acceleration_limits=None,
    figure_size_inches=[8, 10.5],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
):
    if velocity_limits is None:
        velocity_min = np.min([
            df_features['x_velocity_smoothed'].min(),
            df_features['y_velocity_smoothed'].min()
        ])
        velocity_max = np.max([
            df_features['x_velocity_smoothed'].max(),
            df_features['y_velocity_smoothed'].max()
        ])
        velocity_limits = [velocity_min, velocity_max]
    if acceleration_limits is None:
        acceleration_min = np.min([
            df_features['x_acceleration_normalized'].min(),
            df_features['y_acceleration_normalized'].min(),
            df_features['z_acceleration_normalized'].min(),
        ])
        acceleration_max = np.max([
            df_features['x_acceleration_normalized'].max(),
            df_features['y_acceleration_normalized'].max(),
            df_features['z_acceleration_normalized'].max(),
        ])
        acceleration_limits = [acceleration_min, acceleration_max]
    for device_id in df_features['device_id'].unique().tolist():
        df_position_reduced = df_position[df_position['device_id'] == device_id]
        df_features_reduced = df_features[df_features['device_id'] == device_id]
        device_serial_numbers = df_position_reduced['device_serial_number'].unique(
        ).tolist()
        if len(device_serial_numbers) == 0:
            raise ValueError('Device serial number for device ID {} not found in position data'.format(
                device_id
            ))
        if len(device_serial_numbers) > 1:
            raise ValueError('Multiple device serial numbers found in data for device ID {}'.format(
                device_id
            ))
        device_serial_number = device_serial_numbers[0]
        material_names = df_position_reduced['material_name'].unique().tolist()
        if len(material_names) == 0:
            raise ValueError('Material name for device ID {} not found in position data'.format(
                device_id
            ))
        if len(material_names) > 1:
            raise ValueError('Multiple material names found in data for device ID {}'.format(
                device_id
            ))
        entity_name = material_names[0]
        plot_tray_motion_features(
            df_position=df_position_reduced,
            df_features=df_features_reduced,
            entity_name=entity_name,
            device_serial_number=device_serial_number,
            room_corners=room_corners,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            figure_size_inches=figure_size_inches,
            plot_show=plot_show,
            plot_save=plot_save,
            output_directory=output_directory,
            filename_extension=filename_extension,
        )


def plot_tray_motion_features_with_ground_truth_multiple_devices(
    df_position,
    df_features,
    color_dict,
    room_corners=None,
    velocity_limits=None,
    acceleration_limits=None,
    figure_size_inches=[8, 10.5],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
):
    if velocity_limits is None:
        velocity_min = np.min([
            df_features['x_velocity_smoothed'].min(),
            df_features['y_velocity_smoothed'].min()
        ])
        velocity_max = np.max([
            df_features['x_velocity_smoothed'].max(),
            df_features['y_velocity_smoothed'].max()
        ])
        velocity_limits = [velocity_min, velocity_max]
    if acceleration_limits is None:
        acceleration_min = np.min([
            df_features['x_acceleration_normalized'].min(),
            df_features['y_acceleration_normalized'].min(),
            df_features['z_acceleration_normalized'].min(),
        ])
        acceleration_max = np.max([
            df_features['x_acceleration_normalized'].max(),
            df_features['y_acceleration_normalized'].max(),
            df_features['z_acceleration_normalized'].max(),
        ])
        acceleration_limits = [acceleration_min, acceleration_max]
    for device_id in df_features['device_id'].unique().tolist():
        df_position_reduced = df_position[df_position['device_id'] == device_id]
        df_features_reduced = df_features[df_features['device_id'] == device_id]
        device_serial_numbers = df_position_reduced['device_serial_number'].unique(
        ).tolist()
        if len(device_serial_numbers) == 0:
            raise ValueError('Device serial number for device ID {} not found in position data'.format(
                device_id
            ))
        if len(device_serial_numbers) > 1:
            raise ValueError('Multiple device serial numbers found in data for device ID {}'.format(
                device_id
            ))
        device_serial_number = device_serial_numbers[0]
        material_names = df_position_reduced['material_name'].unique().tolist()
        if len(material_names) == 0:
            raise ValueError('Material name for device ID {} not found in position data'.format(
                device_id
            ))
        if len(material_names) > 1:
            raise ValueError('Multiple material names found in data for device ID {}'.format(
                device_id
            ))
        entity_name = material_names[0]
        plot_tray_motion_features_with_ground_truth(
            df_position=df_position_reduced,
            df_features=df_features_reduced,
            entity_name=entity_name,
            device_serial_number=device_serial_number,
            color_dict=color_dict,
            room_corners=room_corners,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            figure_size_inches=figure_size_inches,
            plot_show=plot_show,
            plot_save=plot_save,
            output_directory=output_directory,
            filename_extension=filename_extension,
        )


def plot_tray_motion_features_with_state_multiple_devices(
    df_position,
    df_features,
    color_dict,
    state_field_name='ground_truth_state',
    room_corners=None,
    velocity_limits=None,
    acceleration_limits=None,
    figure_size_inches=[8, 10.5],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
):
    if velocity_limits is None:
        velocity_min = np.min([
            df_features['x_velocity_smoothed'].min(),
            df_features['y_velocity_smoothed'].min()
        ])
        velocity_max = np.max([
            df_features['x_velocity_smoothed'].max(),
            df_features['y_velocity_smoothed'].max()
        ])
        velocity_limits = [velocity_min, velocity_max]
    if acceleration_limits is None:
        acceleration_min = np.min([
            df_features['x_acceleration_normalized'].min(),
            df_features['y_acceleration_normalized'].min(),
            df_features['z_acceleration_normalized'].min(),
        ])
        acceleration_max = np.max([
            df_features['x_acceleration_normalized'].max(),
            df_features['y_acceleration_normalized'].max(),
            df_features['z_acceleration_normalized'].max(),
        ])
        acceleration_limits = [acceleration_min, acceleration_max]
    for device_id in df_features['device_id'].unique().tolist():
        df_position_reduced = df_position[df_position['device_id'] == device_id]
        df_features_reduced = df_features[df_features['device_id'] == device_id]
        device_serial_numbers = df_position_reduced['device_serial_number'].unique(
        ).tolist()
        if len(device_serial_numbers) == 0:
            raise ValueError('Device serial number for device ID {} not found in position data'.format(
                device_id
            ))
        if len(device_serial_numbers) > 1:
            raise ValueError('Multiple device serial numbers found in data for device ID {}'.format(
                device_id
            ))
        device_serial_number = device_serial_numbers[0]
        material_names = df_position_reduced['material_name'].unique().tolist()
        if len(material_names) == 0:
            raise ValueError('Material name for device ID {} not found in position data'.format(
                device_id
            ))
        if len(material_names) > 1:
            raise ValueError('Multiple material names found in data for device ID {}'.format(
                device_id
            ))
        entity_name = material_names[0]
        plot_tray_motion_features_with_state(
            df_position=df_position_reduced,
            df_features=df_features_reduced,
            entity_name=entity_name,
            device_serial_number=device_serial_number,
            color_dict=color_dict,
            state_field_name=state_field_name,
            room_corners=room_corners,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits,
            figure_size_inches=figure_size_inches,
            plot_show=plot_show,
            plot_save=plot_save,
            output_directory=output_directory,
            filename_extension=filename_extension,
        )


def plot_positions(
    df,
    entity_name,
    device_serial_number,
    room_corners,
    marker='.',
    alpha=1.0,
    colormap_name='hot_r',
    quality_lims=[0, 10000],
    figure_size_inches=[10.5, 8],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
    y_axis_labels=['$x$ position (meters)', '$y$ position (meters)'],
    color_axis_label='Quality',
    position_column_names=['x_meters', 'y_meters'],
    quality_column_name='quality'
):
    time_min = df.index.min()
    time_max = df.index.max()
    fig, axes = plt.subplots(2, 1, sharex=True)
    plots = [None, None]
    for axis_index in range(2):
        plots[axis_index] = axes[axis_index].scatter(
            df.index.values,
            df[position_column_names[axis_index]],
            marker=marker,
            alpha=alpha,
            c=df[quality_column_name],
            cmap=plt.get_cmap(colormap_name),
            vmin=quality_lims[0],
            vmax=quality_lims[1]
        )
        axes[axis_index].set_ylim(
            room_corners[0][axis_index],
            room_corners[1][axis_index])
        axes[axis_index].set_ylabel(y_axis_labels[axis_index])
    axes[1].set_xlim(time_min, time_max)
    axes[1].set_xlabel('Time (UTC)')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.colorbar(plots[1], ax=axes).set_label(color_axis_label)
    fig.suptitle('{} ({})'.format(
        entity_name,
        device_serial_number
    ))
    fig.set_size_inches(figure_size_inches[0], figure_size_inches[1])
    if plot_show:
        plt.show()
    if plot_save:
        filename = '_'.join([
            'cuwb_positions',
            slugify.slugify(device_serial_number),
            time_min.strftime('%Y%m%d-%H%M%S'),
            time_max.strftime('%Y%m%d-%H%M%S')
        ])
        filename += '.' + filename_extension
        path = os.path.join(
            output_directory,
            filename
        )
        fig.savefig(path)


def plot_positions_topdown(
    df,
    entity_name,
    device_serial_number,
    room_corners,
    marker='.',
    alpha=1.0,
    color_axis='quality',
    colormap_name='hot_r',
    quality_lims=[0, 10000],
    figure_size_inches=[10.5, 8],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
    axis_labels=['$x$ position (meters)', '$y$ position (meters)'],
    color_axis_label='Quality',
    position_column_names=['x_meters', 'y_meters'],
    quality_column_name='quality'
):
    time_min = df.index.min()
    time_max = df.index.max()
    if color_axis == 'quality':
        color_data = df[quality_column_name]
        vmin = quality_lims[0]
        vmax = quality_lims[1]
    elif color_axis == 'time':
        color_data = df.index
        vmin = time_min
        vmax = time_max
    else:
        raise ValueError('Color axis specification not recognized')
    fig, axes = plt.subplots(1, 1)
    plot = axes.scatter(
        df[position_column_names[0]],
        df[position_column_names[1]],
        marker=marker,
        alpha=alpha,
        c=color_data,
        cmap=plt.get_cmap(colormap_name)
        # vmin=vmin,
        # vmax=vmax
    )
    axes.set_xlim(room_corners[0][0], room_corners[1][0])
    axes.set_ylim(room_corners[0][1], room_corners[1][1])
    axes.set_aspect('equal')
    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    cbar = fig.colorbar(plot, ax=axes)
    cbar.set_label(color_axis_label)
    fig.suptitle('{} ({})'.format(
        entity_name,
        device_serial_number
    ))
    fig.set_size_inches(figure_size_inches[0], figure_size_inches[1])
    fig.autofmt_xdate()
    if plot_show:
        plt.show()
    if plot_save:
        filename = '_'.join([
            'cuwb_positions_topdown',
            slugify.slugify(device_serial_number),
            time_min.strftime('%Y%m%d-%H%M%S'),
            time_max.strftime('%Y%m%d-%H%M%S')
        ])
        filename += '.' + filename_extension
        path = os.path.join(
            output_directory,
            filename
        )
        fig.savefig(path)


def plot_accelerations(
    df,
    entity_name,
    device_serial_number,
    marker='.',
    acceleration_max=2.0,
    alpha=1.0,
    figure_size_inches=[10.5, 8],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
    y_axis_labels=[
        '$x$ acceleration (g\'s)',
        '$y$ acceleration (g\'s)',
        '$z$ acceleration (g\'s)'],
    acceleration_column_names=['x_gs', 'y_gs', 'z_gs']
):
    time_min = df.index.min()
    time_max = df.index.max()
    fig, axes = plt.subplots(3, 1, sharex=True)
    plots = [None, None, None]
    for axis_index in range(3):
        plots[axis_index] = axes[axis_index].scatter(
            df.index.values,
            df[acceleration_column_names[axis_index]],
            marker=marker,
            alpha=alpha
        )
        axes[axis_index].set_ylim(-acceleration_max, acceleration_max)
        axes[axis_index].set_ylabel(y_axis_labels[axis_index])
    axes[2].set_xlim(time_min, time_max)
    axes[2].set_xlabel('Time (UTC)')
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.suptitle('{} ({})'.format(
        entity_name,
        device_serial_number
    ))
    fig.set_size_inches(figure_size_inches[0], figure_size_inches[1])
    if plot_show:
        plt.show()
    if plot_save:
        filename = '_'.join([
            'cuwb_accelerations',
            slugify.slugify(device_serial_number),
            time_min.strftime('%Y%m%d-%H%M%S'),
            time_max.strftime('%Y%m%d-%H%M%S')
        ])
        filename += '.' + filename_extension
        path = os.path.join(
            output_directory,
            filename
        )
        fig.savefig(path)


def plot_positions_and_accelerations(
    df_position,
    df_acceleration,
    entity_name,
    device_serial_number,
    room_corners,
    marker='.',
    alpha=1.0,
    colormap_name='hot_r',
    quality_lims=[0, 10000],
    acceleration_max=2.0,
    figure_size_inches=[8, 10.5],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
    position_y_axis_labels=['$x$ position (meters)', '$y$ position (meters)'],
    position_color_axis_label='Position quality',
    position_column_names=['x_meters', 'y_meters'],
    position_quality_column_name='quality',
    acceleration_y_axis_labels=[
        '$x$ acceleration (g\'s)',
        '$y$ acceleration (g\'s)',
        '$z$ acceleration (g\'s)'],
    acceleration_column_names=['x_gs', 'y_gs', 'z_gs']
):
    time_min = np.min([df_position.index.min(), df_acceleration.index.min()])
    time_max = np.max([df_position.index.max(), df_acceleration.index.max()])
    fig, axes = plt.subplots(5, 1, sharex=True)
    plots = [None, None, None, None, None]
    for position_axis_index in range(2):
        plots[position_axis_index] = axes[position_axis_index].scatter(
            df_position.index.values,
            df_position[position_column_names[position_axis_index]],
            marker=marker,
            alpha=alpha,
            c=df_position[position_quality_column_name],
            cmap=plt.get_cmap(colormap_name),
            vmin=quality_lims[0],
            vmax=quality_lims[1]
        )
        axes[position_axis_index].set_ylim(
            room_corners[0][position_axis_index],
            room_corners[1][position_axis_index])
        axes[position_axis_index].set_ylabel(
            position_y_axis_labels[position_axis_index])
    for acceleration_axis_index in range(3):
        plots[acceleration_axis_index + 2] = axes[acceleration_axis_index + 2].scatter(
            df_acceleration.index.values,
            df_acceleration[acceleration_column_names[acceleration_axis_index]],
            marker=marker,
            alpha=alpha
        )
        axes[acceleration_axis_index +
             2].set_ylim(-acceleration_max, acceleration_max)
        axes[acceleration_axis_index +
             2].set_ylabel(acceleration_y_axis_labels[acceleration_axis_index])
    axes[4].set_xlim(time_min, time_max)
    axes[4].set_xlabel('Time (UTC)')
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.colorbar(plots[1], ax=axes).set_label(position_color_axis_label)
    fig.suptitle('{} ({})'.format(
        entity_name,
        device_serial_number
    ))
    fig.set_size_inches(figure_size_inches[0], figure_size_inches[1])
    if plot_show:
        plt.show()
    if plot_save:
        filename = '_'.join([
            'cuwb_positions_and_accelerations',
            slugify.slugify(device_serial_number),
            time_min.strftime('%Y%m%d-%H%M%S'),
            time_max.strftime('%Y%m%d-%H%M%S')
        ])
        filename += '.' + filename_extension
        path = os.path.join(
            output_directory,
            filename
        )
        fig.savefig(path)


def plot_tray_motion_features(
    df_position,
    df_features,
    entity_name,
    device_serial_number,
    room_corners=None,
    velocity_limits=None,
    acceleration_limits=None,
    figure_size_inches=[8, 10.5],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
):
    time_min = np.min([df_position.index.min(), df_features.index.min()])
    time_max = np.max([df_position.index.max(), df_features.index.max()])
    fig, axes = plt.subplots(7, 1, sharex=True)
    extra_artists = []
    for axis_index, axis_name in enumerate(['x', 'y']):
        axes[axis_index].plot(
            df_position.index,
            df_position['{}_meters'.format(axis_name)],
            'g-',
            label='${}$ position (m)'.format(axis_name)
        )
        axes[axis_index].set_ylabel(r'${}$ (m)'.format(axis_name))
        if room_corners is not None:
            axes[axis_index].set_ylim(
                room_corners[0][axis_index],
                room_corners[1][axis_index]
            )
    for axis_index, axis_name in enumerate(['x', 'y']):
        axes[2 + axis_index].plot(
            df_features.index,
            df_features['{}_velocity_smoothed'.format(axis_name)],
            'b-',
            label='Smoothed ${}$ velocity (m/s)'.format(axis_name)
        )
        axes[2 + axis_index].set_ylabel(r'$d{}/dt$ (m/s)'.format(axis_name))
        if velocity_limits is not None:
            axes[2 + axis_index].set_ylim(
                velocity_limits[0],
                velocity_limits[1]
            )
    for axis_index, axis_name in enumerate(['x', 'y', 'z']):
        axes[4 + axis_index].plot(
            df_features.index,
            df_features['{}_acceleration_normalized'.format(axis_name)],
            'b-',
            label=r'Normalized ${}$ acceleration ($\mathrm{{m}}/\mathrm{{s}}^2$)'.format(
                axis_name)
        )
        axes[4 + axis_index].set_ylabel(
            r'$d^2{}/dt^2$ ($\mathrm{{m}}/\mathrm{{s}}^2$)'.format(axis_name))
        if acceleration_limits is not None:
            axes[4 + axis_index].set_ylim(
                acceleration_limits[0],
                acceleration_limits[1]
            )
    axes[6].set_xlabel('Time (UTC)')
    extra_artists.append(fig.suptitle('{} ({})'.format(
        entity_name,
        device_serial_number
    )))
    fig.set_size_inches(figure_size_inches[0], figure_size_inches[1])
    if plot_show:
        plt.show()
    if plot_save:
        filename = '_'.join([
            'tray_motion_features',
            slugify.slugify(device_serial_number),
            time_min.strftime('%Y%m%d-%H%M%S'),
            time_max.strftime('%Y%m%d-%H%M%S')
        ])
        filename += '.' + filename_extension
        path = os.path.join(
            output_directory,
            filename
        )
        fig.savefig(
            path,
            bbox_extra_artists=extra_artists,
            bbox_inches='tight')


def plot_tray_motion_features_with_ground_truth(
    df_position,
    df_features,
    entity_name,
    device_serial_number,
    color_dict,
    room_corners=None,
    velocity_limits=None,
    acceleration_limits=None,
    figure_size_inches=[8, 10.5],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
):
    time_min = np.min([df_position.index.min(), df_features.index.min()])
    time_max = np.max([df_position.index.max(), df_features.index.max()])
    ground_truth_states = df_features['ground_truth_state'].unique().tolist()
    fig, axes = plt.subplots(7, 1, sharex=True)
    extra_artists = []
    for axis_index, axis_name in enumerate(['x', 'y']):
        axes[axis_index].scatter(
            x=df_position.index,
            y=df_position['{}_meters'.format(axis_name)],
            marker='.',
            s=1,
        )
        axes[axis_index].set_ylabel(r'${}$ (m)'.format(axis_name))
        if room_corners is not None:
            axes[axis_index].set_ylim(
                room_corners[0][axis_index],
                room_corners[1][axis_index]
            )
    for axis_index, axis_name in enumerate(['x', 'y']):
        for ground_truth_state in ground_truth_states:
            axes[2 + axis_index].scatter(
                x=df_features.loc[df_features['ground_truth_state']
                                  == ground_truth_state].index,
                y=df_features.loc[df_features['ground_truth_state'] ==
                                  ground_truth_state, '{}_velocity_smoothed'.format(axis_name)],
                c=color_dict[ground_truth_state],
                marker='.',
                s=1,
                label=ground_truth_state
            )
        axes[2 + axis_index].set_ylabel(r'$d{}/dt$ (m/s)'.format(axis_name))
        if velocity_limits is not None:
            axes[2 + axis_index].set_ylim(
                velocity_limits[0],
                velocity_limits[1]
            )
    for axis_index, axis_name in enumerate(['x', 'y', 'z']):
        for ground_truth_state in ground_truth_states:
            axes[4 + axis_index].scatter(
                x=df_features.loc[df_features['ground_truth_state']
                                  == ground_truth_state].index,
                y=df_features.loc[df_features['ground_truth_state'] ==
                                  ground_truth_state, '{}_acceleration_normalized'.format(axis_name)],
                c=color_dict[ground_truth_state],
                marker='.',
                s=1,
                label=ground_truth_state
            )
        axes[4 + axis_index].set_ylabel(
            r'$d^2{}/dt^2$ ($\mathrm{{m}}/\mathrm{{s}}^2$)'.format(axis_name))
        if acceleration_limits is not None:
            axes[4 + axis_index].set_ylim(
                acceleration_limits[0],
                acceleration_limits[1]
            )
    axes[6].set_xlim(time_min, time_max)
    axes[6].set_xlabel('Time (UTC)')
    extra_artists.append(fig.suptitle('{} ({})'.format(
        entity_name,
        device_serial_number
    )))
    extra_artists.append(
        axes[2].legend(
            loc='upper left',
            bbox_to_anchor=(
                1.0,
                1.0)))
    fig.set_size_inches(figure_size_inches[0], figure_size_inches[1])
    if plot_show:
        plt.show()
    if plot_save:
        filename = '_'.join([
            'tray_motion_features_with_ground_truth',
            slugify.slugify(device_serial_number),
            time_min.strftime('%Y%m%d-%H%M%S'),
            time_max.strftime('%Y%m%d-%H%M%S')
        ])
        filename += '.' + filename_extension
        path = os.path.join(
            output_directory,
            filename
        )
        fig.savefig(
            path,
            bbox_extra_artists=extra_artists,
            bbox_inches='tight')


def plot_tray_motion_features_with_state(
    df_position,
    df_features,
    entity_name,
    device_serial_number,
    color_dict,
    state_field_name='ground_truth_state',
    room_corners=None,
    velocity_limits=None,
    acceleration_limits=None,
    figure_size_inches=[8, 10.5],
    plot_show=True,
    plot_save=False,
    output_directory='.',
    filename_extension='png',
):
    time_min = np.min([df_position.index.min(), df_features.index.min()])
    time_max = np.max([df_position.index.max(), df_features.index.max()])
    states = df_features[state_field_name].unique().tolist()
    fig, axes = plt.subplots(7, 1, sharex=True)
    extra_artists = []
    for axis_index, axis_name in enumerate(['x', 'y']):
        axes[axis_index].scatter(
            x=df_position.index,
            y=df_position['{}_meters'.format(axis_name)],
            marker='.',
            s=1,
        )
        axes[axis_index].set_ylabel(r'${}$ (m)'.format(axis_name))
        if room_corners is not None:
            axes[axis_index].set_ylim(
                room_corners[0][axis_index],
                room_corners[1][axis_index]
            )
    for axis_index, axis_name in enumerate(['x', 'y']):
        for state in states:
            axes[2 + axis_index].scatter(
                x=df_features.loc[df_features[state_field_name]
                                  == state].index,
                y=df_features.loc[df_features[state_field_name] ==
                                  state, '{}_velocity_smoothed'.format(axis_name)],
                c=color_dict[state],
                marker='.',
                s=1,
                label=state
            )
        axes[2 + axis_index].set_ylabel(r'$d{}/dt$ (m/s)'.format(axis_name))
        if velocity_limits is not None:
            axes[2 + axis_index].set_ylim(
                velocity_limits[0],
                velocity_limits[1]
            )
    for axis_index, axis_name in enumerate(['x', 'y', 'z']):
        for state in states:
            axes[4 + axis_index].scatter(
                x=df_features.loc[df_features[state_field_name]
                                  == state].index,
                y=df_features.loc[df_features[state_field_name] ==
                                  state, '{}_acceleration_normalized'.format(axis_name)],
                c=color_dict[state],
                marker='.',
                s=1,
                label=state
            )
        axes[4 + axis_index].set_ylabel(
            r'$d^2{}/dt^2$ ($\mathrm{{m}}/\mathrm{{s}}^2$)'.format(axis_name))
        if acceleration_limits is not None:
            axes[4 + axis_index].set_ylim(
                acceleration_limits[0],
                acceleration_limits[1]
            )
    axes[6].set_xlim(time_min, time_max)
    axes[6].set_xlabel('Time (UTC)')
    extra_artists.append(fig.suptitle('{} ({})'.format(
        entity_name,
        device_serial_number
    )))
    extra_artists.append(
        axes[2].legend(
            loc='upper left',
            bbox_to_anchor=(
                1.0,
                1.0)))
    fig.set_size_inches(figure_size_inches[0], figure_size_inches[1])
    if plot_show:
        plt.show()
    if plot_save:
        filename = '_'.join([
            'tray_motion_features_with_state',
            slugify.slugify(device_serial_number),
            time_min.strftime('%Y%m%d-%H%M%S'),
            time_max.strftime('%Y%m%d-%H%M%S')
        ])
        filename += '.' + filename_extension
        path = os.path.join(
            output_directory,
            filename
        )
        fig.savefig(
            path,
            bbox_extra_artists=extra_artists,
            bbox_inches='tight')
