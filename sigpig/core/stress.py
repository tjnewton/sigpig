"""
Functions to perform stress inversions on slip vectors.
"""

import obspy
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
from obspy.clients.fdsn import Client
import calendar

def stress_Inversion(data, labels) -> Tuple:
    """performs a 1D stress inversion on clustered data. data is x, y coordinates of all events.
    labels is output from cluster() that assigns each event a cluster id. uses the vavrycuk method
    adaptation of michaels method. this implementation performs a separate inversion for each
    cluster. no damping is implemented yet #FIXME."""
    # first extract cluster labels and initialize results list
    unique_Labels = np.unique(labels)
    results = []
    axes = []
    for i in range(len(unique_Labels)):
        print(f"Inverting cluster {i+1} of {len(unique_Labels)}")
        focal_Mechs = data[np.where(labels == i), :][0]
        # modified vavrycuk method
        inversion_Results, stress_Axes = invert(focal_Mechs)
        results.append(inversion_Results)
        axes.append(stress_Axes)
        sigma1_azimuth = round(inversion_Results['sigma_1']['azimuth'], 1)
        sigma1_plunge = round(inversion_Results['sigma_1']['plunge'], 1)
        print(f'Sigma 1     azimuth: {sigma1_azimuth}     plunge: {sigma1_plunge}')
        sigma2_azimuth = round(inversion_Results['sigma_2']['azimuth'], 1)
        sigma2_plunge = round(inversion_Results['sigma_2']['plunge'], 1)
        print(f'Sigma 2     azimuth: {sigma2_azimuth}     plunge: {sigma2_plunge}')
        sigma3_azimuth = round(inversion_Results['sigma_3']['azimuth'], 1)
        sigma3_plunge = round(inversion_Results['sigma_3']['plunge'], 1)
        print(f'Sigma 3     azimuth: {sigma3_azimuth}     plunge: {sigma3_plunge}')
    return (results, axes)

def ellipsoidal_Distance(point1: List[float], point2: List[float], two_dimensional=False) -> float:
    """Returns the distance between two points within an ellipsoidal reference frame
    (WGS-84) for 3 dimensions. Uses geopy's implementation of the
    Karney (2013, https://doi.org/10.1007%2Fs00190-012-0578-z) geodesic algorithm.

    Inputs: IN DEGREES
            point1: list[longitude, latitude, depth]
            point2: list[longitude, latitude, depth]
    """
    longitude1, latitude1, depth1 = point1
    longitude2, latitude2, depth2 = point2
    point1_object = Point(latitude1, longitude1)
    point2_object = Point(latitude2, longitude2)
    distance_2D = distance(point1_object, point2_object).km
    if two_dimensional: # return 2D distance if two_dimensional flag is True
        return abs(distance_2D)
    distance_3D = math.sqrt(distance_2D**2 + ((depth1) - (depth2))**2)
    return abs(distance_3D)

def slab_Figure(SLAB_DATA, scatterData=False):
    """
    scatterData = [ [longitude], [latitude], [depth], 'data_name' ]
    """
    # define the coordinates
    X = SLAB_DATA[:, 0]
    Y = SLAB_DATA[:, 1]
    Z = SLAB_DATA[:, 2]

    # First plot is Scatter3D of data that comprises the slab
    fig = plt.figure(figsize=(10, 6))
    ax = Axes3D(fig)
    cmap = matplotlib.cm.cividis.reversed()
    im = ax.scatter3D(X, Y, Z, s=5, c=Z, cmap=cmap, label="Slab2 data")

    # plot data if flag is True
    if scatterData:
        # define the colors you want to plot with
        colors = itertools.cycle(["grey", "red", "black"])
        for group in scatterData:
            # plot the data
            ax.scatter3D(group[0], group[1], group[2], s=10, c=next(colors),  label=group[3])
    # adjust the view, degrees from x-y plane, rotation of x axis (+ or - 180
    ax.view_init(30, -180)
    # format the colorbar
    fig.colorbar(im)
    ax.set_zlim(-60, 0)

    # set a legend
    legend = ax.legend(fancybox=True, loc='lower left', framealpha=0.75)
    legend.legendPatch.set_facecolor('wheat')
    # show the plot
    plt.show()


    # # Next, a non-working figure of the slab + coastlines & land from cartopy
    # # builds on the mpl hacks found here : https://stackoverflow.com/questions/48269014/contourf-in-3d-cartopy
    # # original working function modified from link in scratch11.py
    # fig = plt.figure()
    # ax3d = fig.add_axes([0, 0, 1, 1], projection='3d')
    #
    # cmap = matplotlib.cm.cividis.reversed()
    # #im = ax3d.scatter3D(X, Y, Z, s=5, c=Z, cmap=cmap, transform=ccrs.PlateCarree())
    #
    # # Make an axes that we can use for mapping the data in 2d.
    # proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=ccrs.Mercator())
    #
    # collection = proj_ax.scatter(X, Y, Z, transform=ccrs.PlateCarree(), alpha=0.4, cmap=cmap)
    # paths = collection.get_paths()
    # # Figure out the matplotlib transform to take us from the X, Y coordinates
    # # to the projection coordinates.
    # trans_to_proj = collection.get_transform() - proj_ax.transData
    #
    # paths = [trans_to_proj.transform_path(path) for path in paths]
    # verts3d = [np.concatenate([path.vertices,
    #                            np.tile(zlev, [path.vertices.shape[0], 1])],
    #                           axis=1)
    #            for path in paths]
    # codes = [path.codes for path in paths]
    # pc = Poly3DCollection([])
    # pc.set_verts_and_codes(verts3d, codes)
    #
    # # Copy all of the parameters from the contour (like colors) manually.
    # # Ideally we would use update_from, but that also copies things like
    # # the transform, and messes up the 3d plot.
    # pc.set_facecolor(collection.get_facecolor())
    # pc.set_edgecolor(collection.get_edgecolor())
    # pc.set_alpha(collection.get_alpha())
    #
    # ax3d.add_collection3d(pc)
    #
    # proj_ax.autoscale_view()
    #
    # ax3d.set_xlim(*proj_ax.get_xlim())
    # ax3d.set_ylim(*proj_ax.get_ylim())
    # ax3d.set_zlim(Z.min(), Z.max())
    #
    # # Now add coastlines.
    # concat = lambda iterable: list(itertools.chain.from_iterable(iterable))
    #
    # target_projection = proj_ax.projection
    #
    # feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')  # 10m, 50m, or 110m
    # geoms = feature.geometries()
    #
    # # Use the convenience (private) method to get the extent as a shapely geometry.
    # boundary = proj_ax._get_extent_geom()
    #
    # # Transform the geometries from PlateCarree into the desired projection.
    # geoms = [target_projection.project_geometry(geom, feature.crs)
    #          for geom in geoms]
    # # Clip the geometries based on the extent of the map (because mpl3d can't do it for us)
    # geoms = [boundary.intersection(geom.buffer(0)) for geom in geoms]
    #
    # # Convert the geometries to paths so we can use them in matplotlib.
    # paths = concat(geos_to_path(geom) for geom in geoms)
    # polys = concat(path.to_polygons() for path in paths)
    # lc = PolyCollection(polys, edgecolor='black',
    #                     facecolor='green', closed=True)
    # lc.set_alpha(0.4)
    # ax3d.add_collection3d(lc, zs=ax3d.get_zlim()[0])
    #
    # plt.close(proj_ax.figure)
    #
    # # adjust the view
    # ax3d.view_init(30, -120)
    # plt.show()
    #
    #
    #
    #
    #
    #
    # # # wrap it up and merge with slab data
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # ax.set_zlabel('Height')
    # # plt.show()

def geospatial_Figure(title: str, extent: list, scatterData=[], clusterData=(), principalStressData=(), displayLegend=False):
    """Produces a map optionally displaying geospatial data

    title = "Seismicity filtered by distance from megathrust"
    extent = [130, 138, 30, 38]
    scatterData = [ [longitude], [latitude], 'data_name' ]

    # FIXME add in tectonic plates. plot shapefiles via cartopy? could also try this type
    # of implementation for slabs. Potential plates data source:
    # https://github.com/fraxen/tectonicplates

    """
    # Define the coordinate system of the data
    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
    # Create a Stamen map tiler instance, and use its CRS for the GeoAxes.
    tiler = Stamen('terrain-background')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=tiler.crs)
    ax.set_title(title)
    # Pick the area of interest
    ax.set_extent(extent, geodetic)
    # Add the Stamen aerial imagery at zoom level 7
    ax.add_image(tiler, 7)

    # plot data of various types if it exists
    if len(scatterData) > 0:
        # define the colors you want to plot with
        colors = itertools.cycle(["black", "red"])
        #colors = itertools.cycle(["black", "black"])
        for group in scatterData:
            # plot the data
            ax.scatter(group[0], group[1],
                       c=next(colors), linewidth=0.1, marker='o',
                       alpha=0.4, transform=ccrs.PlateCarree(),
                       label=group[2])

            # plot cluster boundaries
            # plt.plot(NIED_CLUSTER_DATA[:,0], NIED_CLUSTER_DATA[:,1],
            #         color='gray', linewidth=2, linestyle='--',
            #         transform=ccrs.Geodetic(),
            #         )

        # Create a legend for the data
        #legend_artists = [Line([0], [0], color=color, linewidth=3)
        #                  for color in ('white', 'gray')]
        #legend_texts = ['Correct ellipse\n(WGS84)', 'Incorrect ellipse\n(sphere)']
        #legend = ax.legend(legend_artists, legend_texts, fancybox=True,
        #                   loc='lower right', framealpha=0.75)
        if displayLegend:
            legend = ax.legend(fancybox=True, loc='lower right', framealpha=0.75)
            legend.legendPatch.set_facecolor('wheat')

    if len(clusterData) > 0:
        cluster_Points, cluster_Borders = clusterData

        borders = []
        colors = []
        for index, group in enumerate(cluster_Borders):
            # assemble the polygons that represent the borders, then plot them
            borders.append(Polygon(group, True))
            colors.append(index)
        p = PatchCollection(borders, transform=ccrs.PlateCarree(), alpha=0.4)
        p.set(array=np.array(colors), cmap='plasma') # set color of patch
        ax.add_collection(p)
        #p.set_array(np.array([colors]))

        #color = iter(plt.cm.rainbow(np.linspace(0, 1, len(cluster_Points))))
        cmap = matplotlib.cm.get_cmap('plasma') # FIXME weird way to assign colors but it works
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(colors)-1)
        for index, group in enumerate(cluster_Points):
            # plot the data
            ax.scatter(group[:,0], group[:,1],
                       linewidth=0.1, marker='o',
                       alpha=0.4, transform=ccrs.PlateCarree(),
                       c=np.array(cmap(norm(colors[index]))).reshape(1,4))

    if len(principalStressData) > 0:
        cluster_Points, cluster_Borders, sigma_1_plunge, centers = principalStressData

        borders = []
        colors = []
        for index, group in enumerate(cluster_Borders):
            # assemble the polygons that represent the borders, then plot them
            borders.append(Polygon(group, True))
            colors.append(sigma_1_plunge[index])
        p = PatchCollection(borders, transform=ccrs.PlateCarree(), alpha=0.4)
        p.set(array=np.array(colors), cmap='plasma') # set color of patch
        ax.add_collection(p)
        #p.set_array(np.array([colors]))

        #color = iter(plt.cm.rainbow(np.linspace(0, 1, len(cluster_Points))))
        #cmap = matplotlib.cm.get_cmap('plasma') # FIXME weird way to assign colors but it works
        #norm = matplotlib.colors.Normalize(vmin=np.amin(np.array(colors)), vmax=np.amax(np.array(colors)))
        for index, group in enumerate(cluster_Points):
            # plot the data
            # a= ax.scatter(group[:,0], group[:,1],
            #            linewidth=0.1, marker='o',
            #            alpha=0.4, transform=ccrs.PlateCarree(),
            #            c=np.array(cmap(norm(colors[index]))).reshape(1,4))

            # FIXME check that colorbar is consistent with exact patch values
            #c = [colors[index] for item in group[:, 0]]
            c = [sigma_1_plunge[index] for item in group[:, 0]]
            vmin = np.amin(sigma_1_plunge)
            vmax = np.amax(sigma_1_plunge)
            a = ax.scatter(group[:, 0], group[:, 1], c=c,
                       cmap="plasma", linewidth=0.1, marker='o',
                       alpha=0.4, transform=ccrs.PlateCarree(),
                       vmin=vmin, vmax=vmax)
            ax.text(centers[index][0], centers[index][1], str(index), fontsize=20, transform=ccrs.PlateCarree())
        plt.colorbar(a)
    #fig.colorbar(p)

    # Create an inset GeoAxes showing the location of the map area
    sub_ax = fig.add_axes([0.24, 0.68, 0.2, 0.2],
                          projection=ccrs.PlateCarree())
    # check the extent of the inset FIXME ::: this should be set by some scaling factor
    sub_ax.set_extent([extent[0] - 10, extent[1] + 10, extent[2] - 10, extent[3] + 10], geodetic)
    # Make a nice border around the inset axes.
    effect = Stroke(linewidth=4, foreground='wheat', alpha=0.5)
    sub_ax.spines['geo'].set_path_effects([effect])

    # Add the land, coastlines and the extent of the map area
    sub_ax.add_feature(cfeature.LAND)
    sub_ax.coastlines()

    extent_box = sgeom.box(extent[0], extent[2], extent[1], extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), facecolor='none',
                          edgecolor='blue', linewidth=2)
    plt.show()

def interface_Fliter(dataset, boundary: str, distance_From_Boundary: float, depth_Filter=False, interpolation=False, figure=False):
    """returns all events in dataset that are within plus or minus (inclusive)
    the specified distance of the specified tectonic boundary file, boundary.

    Don't approach the edges of the specified boundary. This function does not
    incorporate clipping so values will be unreasonable at the edges due to
    extrapolation onto a gridded reference.

    dataset format: ndarray N x 3, or name of preloaded dataset as string
    boundary format: string of slab name, e.g. Ryukyu, or Cascadia
    distance: float of distance from tectonic boundary
    """
    # load specified .xyz file
    # FIXME this needs to dynamically reference slabs other than Ryukyu & Cascadia
    # Use a faster distance calculation like: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html#scipy.interpolate.NearestNDInterpolator
    if boundary == "Ryukyu":
        SLAB_DATA = np.genfromtxt(
            '/Users/human/Dropbox/Programs/slabInspector/slab_models/slab2/Ryukyu/ryu_slab2_dep_02.26.18.xyz',
            delimiter=',')
        SLAB_MASK = np.genfromtxt(
            '/Users/human/Dropbox/Programs/slabInspector/slab_models/slab2/Ryukyu/ryu_slab2_clp_02.26.18.csv',
            delimiter=',')

    elif boundary == "Cascadia":
        SLAB_DATA = np.genfromtxt(
            '/Users/human/Dropbox/Programs/slabInspector/slab_models/slab2/Cascadia/cas_slab2_dep_02.24.18.xyz',
            delimiter=',')
        SLAB_MASK = np.genfromtxt(
            '/Users/human/Dropbox/Programs/slabInspector/slab_models/slab2/Cascadia/cas_slab2_clp_02.24.18.csv',
            delimiter=',')

    # remove NaNs
    SLAB_DATA = SLAB_DATA[~np.isnan(SLAB_DATA).any(axis=1)]
    # convert negative depths to positive values
    SLAB_DATA[:,2] = abs(SLAB_DATA[:,2])

    # if depth_Filter is on, filter the data
    if depth_Filter:
        depth_Mask = SLAB_DATA[:, 2] <= depth_Filter
        SLAB_DATA = SLAB_DATA[depth_Mask, :]

    SLAB_DATA_MINS = np.amin(SLAB_DATA, axis=0)
    SLAB_DATA_MAXS = np.amax(SLAB_DATA, axis=0)
    SLAB_DATA_RANGES = SLAB_DATA_MAXS - SLAB_DATA_MINS
    X = SLAB_DATA[:, 0]
    Y = SLAB_DATA[:, 1]
    Z = -1 * SLAB_DATA[:, 2]

    if figure:
        slab_Figure(np.array([X, Y, Z]).T)

    if interpolation:
        # use scipy.interpolate
        # https://scipython.com/book/chapter-8-scipy/examples/scipyinterpolateinterp2d/
        # https://stackoverflow.com/questions/46040382/spline-interpolation-in-3d-in-python
        # https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy
        #interpolation_Function = interp2d(X, Y, Z, kind='cubic') # this line takes 18 minutes and produces bad results
        #interpolation_Function = Rbf(X, Y, Z) # takes 1 minute, but the interpolation below doesn't work
        # FIXME interpolation in it's current state doesn't work. To make it work

        # generate grid for interpolation
        grid_Size = 0.01
        x = np.linspace(SLAB_DATA_MINS[0], SLAB_DATA_MAXS[0],
                        num=int(SLAB_DATA_RANGES[0] / grid_Size),
                        endpoint=True, retstep=False)
        y = np.linspace(SLAB_DATA_MINS[1], SLAB_DATA_MAXS[1],
                        num=int(SLAB_DATA_RANGES[1] / grid_Size),
                        endpoint=True, retstep=False)
        #z = interpolation_Function(x, y) # lowercase z values are interpolated. uppercase Z is slab2.0 points, used for interp2d

        x_Mesh, y_Mesh = np.meshgrid(x, y)
        z = griddata(np.array([X, Y]).T, Z, (x_Mesh, y_Mesh), method='cubic')  # xy points, values, xy grid to interpolate, method

        # visualize masked interpolated slab to check for errors
        # # clipping isn't needed with griddata (it doesn't extrapolate, stops at convex hull)
        # SLAB_MASK = (SLAB_MASK - SLAB_DATA_MINS[0:2]) / grid_Size # convert mask coordinates to grid coordinates
        # # change values < 0 to 0
        # SLAB_MASK[SLAB_MASK < 0] = 0
        # SLAB_MASK = list(map(tuple, SLAB_MASK))
        # SLAB_MASK_IMG = Image.new("L", [len(x), len(y)], 0)
        # ImageDraw.Draw(SLAB_MASK_IMG).polygon(SLAB_MASK, outline=1, fill=1)
        # SLAB_MASK = np.array(SLAB_MASK_IMG)

        # unpack x, y, and z for use with slab_Figure()
        x_List = []
        y_List = []
        z_List = []
        for row_idx in range(len(y)): # 1275
            for col_idx in range(len(x)): # 1865
                # if SLAB_MASK[row_idx][col_idx]: # if masking
                if not np.isnan(z[row_idx, col_idx]): # if value isn't nan
                    x_List.append(x[col_idx])
                    y_List.append(y[row_idx])
                    z_List.append(z[row_idx, col_idx])

        # z = interpolation_Function(x_List, y_List) # rbf interpolation, ends with exit code 137

        if figure:
            slab_Figure(np.array([x_List, y_List, z_List]).T)

    # loop through dataset and slab data to find closest values for comparison
    if type(dataset) == str: # if we are using a previously used dataset
        # conditionals defining previously used datasets and distances
        if dataset == "NIED_XYZ_DATA":
            # load dataset
            dataset = np.genfromtxt(
                '/Users/human/Dropbox/Research/Nankai_Stress/NIED_declustered_catalog/dataset_S1.csv', delimiter=',')
            dataset = np.array(
                [[dataset[i][2], dataset[i][1], dataset[i][3],
                           dataset[i][4], dataset[i][5], dataset[i][6]] for i in range(1, len(dataset))])

            # load pickle file of predetermined distances
            infile = open('NIED_XYZ_DATA_120km_distances', 'rb') # contains events down to 120km
            distances = pickle.load(infile)
            infile.close()

        elif dataset == "CASCADIA_XYZ_DATA":
            # load dataset pickle file
            infile = open('CASCADIA_XYZ_DATA_80km_dataset', 'rb')  # contains events down to 80km
            dataset = pickle.load(infile)
            infile.close()

            # load pickle file of predetermined distances
            infile = open('CASCADIA_XYZ_DATA_80km_distances', 'rb') # contains events down to 80km
            distances = pickle.load(infile)
            infile.close()

        elif dataset == "LFE_XYZ_DATA":
            # load dataset pickle file
            infile = open('LFE_XYZ_DATA_200km_dataset', 'rb')  # contains events down to 80km
            dataset = pickle.load(infile)
            infile.close()

            # load pickle file of predetermined distances
            infile = open('LFE_XYZ_DATA_200km_distances', 'rb') # contains events down to 80km
            distances = pickle.load(infile)
            infile.close()

    else: # for new datasets and distances
        distances = []
        # this loop takes ~3:47 to
        for row_idx, row in enumerate(dataset):
            print(f"Analyzing dataset row {row_idx} of {len(dataset)}")
            distance_To_Slab = np.array([ellipsoidal_Distance([row[0], row[1], row[2]*-1], [X[i], Y[i], Z[i]]) for i in range(len(X))])
            distances.append(np.amin(distance_To_Slab))
        # save distances as pickle file
        outfile = open('SOMETHING_XYZ_DATA_200km_distances', 'wb')
        pickle.dump(distances, outfile)
        outfile.close()
        # save dataset as pickle file
        outfile = open('SOMETHING_XYZ_DATA_200km_dataset', 'wb')
        pickle.dump(dataset, outfile)
        outfile.close()

    distances = np.array(distances)
    # find points that are <= the specified distance from the boundary
    within_Volume = dataset[distances <= distance_From_Boundary]
    outside_Volume = dataset[distances > distance_From_Boundary]

    # plot points inside/outside specified volume relative to slab and land
    if figure:
        slab_Figure(np.array([X, Y, Z]).T,
            scatterData=[ [list(outside_Volume[:,0]), list(outside_Volume[:,1]), list(outside_Volume[:,2]*-1), f"> {distance_From_Boundary} km from megathrust"],
                        [list(within_Volume[:, 0]), list(within_Volume[:, 1]), list(within_Volume[:,2]*-1), f"within {distance_From_Boundary} km of megathrust"]
                        ])
        extent = [np.amin(dataset, axis=0)[0], np.amax(dataset, axis=0)[0], np.amin(dataset, axis=0)[1], np.amax(dataset, axis=0)[1]]
        geospatial_Figure("Seismicity filtered by distance from megathrust", extent,
                   scatterData=[ [list(outside_Volume[:,0]), list(outside_Volume[:,1]), f"> {distance_From_Boundary} km from megathrust"],
                               [list(within_Volume[:, 0]), list(within_Volume[:, 1]), f"within {distance_From_Boundary} km of megathrust"]
                               ], displayLegend=True)

    return within_Volume

def cluster(data: np.array, num_Clusters: int, min_Cluster_Size: int,
        max_Cluster_Size: int, n_init=10, max_iter=300)-> Tuple[np.ndarray, np.ndarray]:
    """
    A wrapper for k_means_constrained clustering
    # FIXME ::: k_means_constrained has other functionality (e.g. predict)
    # that should be included in this wrapper, for now only labels and centers are returned
    """
    # data = np.array([[1, 2], [1, 4], [1, 0],
    #               [4, 2], [4, 4], [4, 0]])
    # num_Clusters = 2
    # min_Cluster_Size = 2
    # max_Cluster_Size = 5

    clusterer = KMeansConstrained(
        n_clusters=num_Clusters,
        size_min=min_Cluster_Size,
        size_max=max_Cluster_Size,
        random_state=0,
        n_init=n_init,
        max_iter=max_iter,
        init="random")
    clusterer.fit(data)
    labels = clusterer.labels_
    centers = clusterer.cluster_centers_
    return labels, centers

def boundary(points: np.ndarray):
    """Returns cluster boundary via SciPy's ConvexHull"""
    hull = ConvexHull(points)
    return hull.vertices

def visualize_Clusters(data: np.array, labels: np.ndarray, extent: list):
    """
    Uses cartopy to visualize clusters generated by cluster function.
    Returns a map of data colored by cluster, with cluster boundaries and centers
    """
    # first extract cluster borders and points
    cluster_Borders = []
    cluster_Points = []
    for i in range(len(np.unique(labels))):
        points = data[np.where(labels == i), :][0]
        cluster_Points.append(points)
        cluster_Borders.append(points[boundary(points), :])
    # now call geospatial_Figure with clusterData
    geospatial_Figure(" ", extent,
                      clusterData=(cluster_Points, cluster_Borders)) # f"{len(np.unique(labels))} clusters : borders and members"

def split_Data(events: np.ndarray, labels: np.ndarray):
    """take in events and corresponding cluster labels, outputs seperate .txt
    files for each cluster in format for FMC kaverina diagram program input

    parameters:
    events - np.ndarray with columns lon, lat, depth, strike, dip, rake,
             magnitude
    labels - 1d array of labels, same length as events
    """
    for i in range(len(np.unique(labels))):
        cluster_Data = events[np.where(labels == i), :][0]
        # check if magnitudes are specified
        if len(cluster_Data[0]) < 7:
            cluster_Data = np.hstack((cluster_Data, np.full([len(cluster_Data),
                                                            1], 5)))
        # construct X, Y, ID array then stack vertically to combine
        x_y_id = np.hstack((np.full([len(cluster_Data),1],'X'), np.full([len(
            cluster_Data),1],'Y'), np.full([len(cluster_Data),1],str(i))))
        cluster_Data = np.hstack((cluster_Data, x_y_id))

        # # save events as pickle if you fancy
        # outfile = open(f'events_{i}', 'wb')
        # pickle.dump(cluster_Data, outfile)
        # outfile.close()

        # save events as text file
        fmt = '%s' #'%f, %f, %f, %f, %f, %f, %f, %s, %s, %f'
        np.savetxt(f'events_{i}.txt', cluster_Data, delimiter=' ', fmt=fmt)

        print(f"Cluster {i} finished.")
    return None

def visualize_Sigma_1(data: np.array, labels: np.ndarray, extent: list, sigma_1_plunge, centers):
    """
    Uses cartopy to visualize clusters and their principal stress axes
    Returns a map of data colored by cluster and sigma_1 plunge
    """
    # first extract cluster borders and points
    cluster_Borders = []
    cluster_Points = []
    # for plotting
    for i in range(len(np.unique(labels))):
        points = data[np.where(labels == i), :][0]
        cluster_Points.append(points)
        cluster_Borders.append(points[boundary(points), :])
    # now call geospatial_Figure with principalStressData
    geospatial_Figure("sigma_1 plunge", extent,
                      principalStressData=(cluster_Points, cluster_Borders, sigma_1_plunge, centers)) # f"{len(np.unique(labels))} clusters : borders and members"

def visualize_Principal_Stresses(stress_Axes: List[Tuple]):
    'driver for Vavrycuk plot of principal stress axes. stress_Axes is returned by stress_Inversion'
    for index, group in enumerate(stress_Axes):
        sigma_vector_1_statistics, sigma_vector_2_statistics, sigma_vector_3_statistics = group
        plot_stress_axes.plot_stress_axes(sigma_vector_1_statistics, sigma_vector_2_statistics, sigma_vector_3_statistics, "dummy_name", index)

